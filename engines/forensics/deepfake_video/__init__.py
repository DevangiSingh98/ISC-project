"""
Video Forensics Engine – Full Implementation (v2)

Analyses video files for:
1. Per-frame pixel-domain manipulation (reuses ManipulationLocalizer)
2. Per-frame AI-generated / synthetic detection (reuses AIGeneratedImageDetector)
3. Temporal inconsistency via Lucas-Kanade optical flow across consecutive sampled frames
4. Face-region deepfake detection using OpenCV Haar cascade + per-frame noise analysis
5. Aggregated deepfake score and suspicious-frame list with thumbnails + heatmaps
"""
import base64
import io
import os
import tempfile
import time
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image

# Re-use existing image engines — these are imported lazily inside the class
# to avoid circular imports and to keep the module importable even when
# torch is not yet warm.

HAAR_FACE_PATH: str = ""
try:
    # cv2.data is available in full opencv builds; may be absent in some
    # headless or minimal builds. Guard gracefully.
    _data_mod = getattr(cv2, "data", None)
    if _data_mod is not None:
        HAAR_FACE_PATH = getattr(_data_mod, "haarcascades", "") + "haarcascade_frontalface_default.xml"
except Exception:
    HAAR_FACE_PATH = ""

# Sampling strategy constants
DEFAULT_SAMPLE_FPS = 1.0        # Analyse 1 frame per second by default
MAX_FRAMES_TO_ANALYSE = 60     # Hard cap to keep latency reasonable
VIDEOMAE_FRAMES = 16
SUSPICIOUS_FRAME_THRESHOLD = 0.40  # Score above which a frame is "suspicious"
THUMBNAIL_SIZE = (160, 90)       # Width × height for strip thumbnails
DEFAULT_VIDEO_MODEL = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "models",
    "celebdf_resnet18", "celebdf_resnet18.pt",
))


def _frame_to_pil(frame_bgr: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR frame to a PIL RGB image."""
    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


def _encode_pil_thumbnail(pil_img: Image.Image, size: Tuple[int, int] = THUMBNAIL_SIZE) -> str:
    """Resize and JPEG-encode a PIL image; return as data-URI base64 string."""
    thumb = pil_img.copy()
    thumb.thumbnail(size, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=70)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def _compute_optical_flow_inconsistency(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    face_roi: Optional[Tuple[int, int, int, int]] = None,
) -> float:
    """
    Computes a temporal inconsistency score between two consecutive frames.

    Uses Lucas-Kanade sparse optical flow on Shi-Tomasi corner points.
    If a face ROI is provided, flow is restricted to that region for a
    more targeted deepfake signal (face-swap / puppet artefacts produce
    unnatural discontinuities).

    Returns a score in [0, 1] where higher = more temporally inconsistent.
    """
    h, w = prev_gray.shape

    # Restrict to face ROI if detected
    if face_roi is not None:
        x, y, fw, fh = face_roi
        x, y = max(0, x), max(0, y)
        x2, y2 = min(w, x + fw), min(h, y + fh)
        region_prev = prev_gray[y:y2, x:x2]
        region_curr = curr_gray[y:y2, x:x2]
    else:
        region_prev = prev_gray
        region_curr = curr_gray

    if region_prev.size == 0 or region_curr.size == 0:
        return 0.0

    # Shi-Tomasi good features to track
    corners = cv2.goodFeaturesToTrack(
        region_prev,
        maxCorners=80,
        qualityLevel=0.01,
        minDistance=5,
        blockSize=5,
    )

    if corners is None or len(corners) < 4:
        # Fallback: frame-level mean absolute difference
        diff = cv2.absdiff(region_prev, region_curr)
        mad = float(np.mean(diff)) / 255.0
        return float(np.clip(mad * 4.0, 0.0, 1.0))

    # Map corners back to full frame coordinates when face ROI is active
    if face_roi is not None:
        x_off, y_off = face_roi[0], face_roi[1]
        full_corners = corners.copy()
        full_corners[:, :, 0] += x_off
        full_corners[:, :, 1] += y_off
    else:
        full_corners = corners

    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        curr_gray,
        full_corners.astype(np.float32),
        None,
        winSize=(15, 15),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    )

    if next_pts is None or status is None:
        return 0.0

    good_prev = full_corners[status.ravel() == 1]
    good_next = next_pts[status.ravel() == 1]

    if len(good_prev) < 2:
        return 0.0

    # Estimate the dominant camera/object motion first. Natural camera pans
    # should be removed before measuring local temporal disagreement.
    prev_points = good_prev.reshape(-1, 1, 2).astype(np.float32)
    next_points = good_next.reshape(-1, 1, 2).astype(np.float32)
    affine, inliers = cv2.estimateAffinePartial2D(
        prev_points,
        next_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.5,
    )
    if affine is not None and inliers is not None and int(np.count_nonzero(inliers)) >= 3:
        predicted = cv2.transform(prev_points, affine).reshape(-1, 2)
        residuals = np.linalg.norm(good_next.reshape(-1, 2) - predicted, axis=1)
        inlier_residuals = residuals[inliers.ravel().astype(bool)]
        if len(inlier_residuals) >= 3:
            # Median residual is robust to a few bad tracks or an occlusion.
            residual = float(np.median(inlier_residuals))
            spread = float(np.median(np.abs(inlier_residuals - residual)))
            return float(np.clip((residual + spread) / 6.0, 0.0, 1.0))

    # If a global motion model cannot be estimated, retain a conservative
    # fallback based on flow dispersion rather than absolute motion.
    flow_vecs = good_next - good_prev
    magnitudes = np.linalg.norm(flow_vecs, axis=2).ravel()
    mean_mag = float(np.mean(magnitudes)) + 1e-6
    cv_score = float(np.std(magnitudes)) / mean_mag
    return float(np.clip(cv_score / 2.5, 0.0, 1.0))


class VideoForensicsDetector:
    """
    Full Video Forensics Detector.

    Analyses an MP4/MOV/WebM video by:
      - Sampling frames at a target FPS
      - Running the existing ManipulationLocalizer + AIGeneratedImageDetector
        on each frame (Track B image engines, reused as-is)
      - Computing inter-frame optical flow to detect temporal inconsistencies
      - Detecting faces with OpenCV Haar cascade and focusing deepfake
        analysis on facial regions
      - Producing an aggregated deepfake score, temporal inconsistency score,
        and a list of suspicious frames with thumbnails and heatmaps
    """

    def __init__(self, sample_fps: float = DEFAULT_SAMPLE_FPS):
        self.sample_fps = sample_fps
        self.version = "v2.1-video-model-ready"

        # Lazy-imported image engines (avoid heavy torch init at module import time)
        self._localizer = None
        self._synthetic_detector = None
        self._face_cascade = None
        self._videomae = None
        self._videomae_status = "not_configured"
        self._video_model_threshold = 0.5
        self._videomae_kind = None

    def _get_localizer(self):
        if self._localizer is None:
            from engines.forensics.manipulation.localizer import ManipulationLocalizer
            self._localizer = ManipulationLocalizer()
        return self._localizer

    def _get_synthetic_detector(self):
        if self._synthetic_detector is None:
            from engines.forensics.synthetic.detector import AIGeneratedImageDetector
            self._synthetic_detector = AIGeneratedImageDetector()
        return self._synthetic_detector

    def _get_face_cascade(self):
        if self._face_cascade is None:
            try:
                if HAAR_FACE_PATH and os.path.exists(HAAR_FACE_PATH):
                    cascade = cv2.CascadeClassifier(HAAR_FACE_PATH)
                    # CascadeClassifier() returns an empty object on failure, not None
                    if not cascade.empty():
                        self._face_cascade = cascade
                    else:
                        self._face_cascade = False  # sentinel: unavailable
                else:
                    self._face_cascade = False
            except (AttributeError, Exception):
                # CascadeClassifier not available in this opencv build
                self._face_cascade = False
        return self._face_cascade

    def _get_videomae(self):
        """Load a trained Celeb-DF frame/temporal checkpoint when configured."""
        if self._videomae is not None or self._videomae_status != "not_configured":
            return self._videomae

        checkpoint = os.environ.get("NEWSPROOF_VIDEO_MODEL", "").strip()
        if not checkpoint:
            checkpoint = os.environ.get("NEWSPROOF_VIDEOMAE_WEIGHTS", "").strip()
        if not checkpoint:
            videomae_100_default = os.path.abspath(os.path.join(
                os.path.dirname(__file__), "..", "..", "..", "models",
                "celebdf_videomae_100", "videomae-celebdf",
            ))
            videomae_default = os.path.abspath(os.path.join(
                os.path.dirname(__file__), "..", "..", "..", "models",
                "celebdf_videomae", "videomae-celebdf",
            ))
            if os.path.isdir(videomae_100_default):
                checkpoint = videomae_100_default
            else:
                checkpoint = videomae_default if os.path.isdir(videomae_default) else DEFAULT_VIDEO_MODEL
        if not checkpoint:
            self._videomae_status = "not_configured"
            return None
        if not os.path.exists(checkpoint):
            self._videomae_status = "checkpoint_missing"
            return None

        try:
            import torch
            if os.path.isdir(checkpoint):
                from transformers import AutoImageProcessor, VideoMAEForVideoClassification
                self._videomae_processor = AutoImageProcessor.from_pretrained(checkpoint)
                self._videomae = VideoMAEForVideoClassification.from_pretrained(checkpoint)
                self._videomae_kind = "videomae"
                report_path = os.path.join(os.path.dirname(checkpoint), "videomae-celebdf.json")
                if os.path.isfile(report_path):
                    import json
                    with open(report_path, "r", encoding="utf-8") as report_file:
                        report = json.load(report_file)
                    self._video_model_threshold = float(
                        report.get("decision_threshold", report.get("metrics", {}).get("decision_threshold", 0.5))
                    )
            else:
                from torchvision.models import resnet18
                import torchvision.transforms as T
                state = torch.load(checkpoint, map_location="cpu")
                model = resnet18(weights=None)
                model.fc = torch.nn.Linear(model.fc.in_features, 2)
                model.load_state_dict(state.get("state_dict", state))
                self._video_model_threshold = float(state.get("decision_threshold", 0.5))
                self._videomae_processor = T.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                )
                self._videomae = model
                self._videomae_kind = "resnet18"
            self._videomae.eval()
            self._videomae_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self._videomae.to(self._videomae_device)
            self._videomae_status = "loaded"
        except (ImportError, OSError, RuntimeError, ValueError):
            self._videomae = None
            self._videomae_status = "load_failed"
        return self._videomae

    def _analyze_videomae(self, frames: List[np.ndarray]) -> Optional[float]:
        """Return the trained frame/temporal model's fake probability."""
        model = self._get_videomae()
        if model is None:
            return None
        try:
            import torch
            selected = np.linspace(0, len(frames) - 1, VIDEOMAE_FRAMES).round().astype(int)
            probabilities = []
            with torch.inference_mode():
                if self._videomae_kind == "videomae":
                    rgb_frames = []
                    for index in selected:
                        frame = frames[int(index)]
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        face_roi = self._detect_largest_face(gray)
                        if face_roi is not None:
                            x, y, fw, fh = face_roi
                            margin = int(0.25 * max(fw, fh))
                            x1, y1 = max(0, x - margin), max(0, y - margin)
                            x2, y2 = min(frame.shape[1], x + fw + margin), min(frame.shape[0], y + fh + margin)
                            crop = frame[y1:y2, x1:x2]
                        else:
                            crop = frame
                        crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
                        rgb_frames.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    inputs = self._videomae_processor(rgb_frames, return_tensors="pt")
                    inputs = {key: value.to(self._videomae_device) for key, value in inputs.items()}
                    logits = model(**inputs).logits
                    return float(torch.softmax(logits, dim=-1)[0, 1].item())

                for index in selected:
                    frame = frames[int(index)]
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    face_roi = self._detect_largest_face(gray)
                    if face_roi is not None:
                        x, y, fw, fh = face_roi
                        margin = int(0.25 * max(fw, fh))
                        x1, y1 = max(0, x - margin), max(0, y - margin)
                        x2, y2 = min(frame.shape[1], x + fw + margin), min(frame.shape[0], y + fh + margin)
                        crop = frame[y1:y2, x1:x2]
                    else:
                        crop = frame
                    crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
                    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
                    tensor = self._videomae_processor(tensor).unsqueeze(0).to(self._videomae_device)
                    logits = model(tensor)
                    probabilities.append(float(torch.softmax(logits, dim=1)[0, 1].item()))
            if not probabilities:
                return None
            mean_probability = float(np.mean(probabilities))
            maximum_probability = float(np.max(probabilities))
            return float(np.clip(0.65 * mean_probability + 0.35 * maximum_probability, 0.0, 1.0))
        except (IndexError, KeyError, RuntimeError, ValueError, TypeError):
            self._videomae_status = "inference_failed"
            return None

    # ------------------------------------------------------------------
    # Frame sampling
    # ------------------------------------------------------------------

    def _select_frame_indices(
        self, total_frames: int, source_fps: float
    ) -> List[int]:
        """
        Selects frame indices to analyse at self.sample_fps.
        Caps the total at MAX_FRAMES_TO_ANALYSE.
        Always includes the first and last frame.
        """
        if source_fps <= 0 or total_frames == 0:
            return []

        step = max(1, int(round(source_fps / self.sample_fps)))
        indices = list(range(0, total_frames, step))

        # Always include last frame
        if total_frames - 1 not in indices:
            indices.append(total_frames - 1)

        # Cap at maximum
        if len(indices) > MAX_FRAMES_TO_ANALYSE:
            # Preserve coverage across the entire video, including its end.
            positions = np.linspace(
                0, len(indices) - 1, MAX_FRAMES_TO_ANALYSE
            ).round().astype(int)
            indices = [indices[position] for position in positions]

        return sorted(set(indices))

    # ------------------------------------------------------------------
    # Face detection
    # ------------------------------------------------------------------

    def _detect_largest_face(
        self, gray: np.ndarray
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Runs Haar face detection. Returns (x, y, w, h) of the largest face,
        or None if no face is found or the cascade is unavailable.
        """
        cascade = self._get_face_cascade()
        # cascade is False when CascadeClassifier is unavailable
        if not cascade:
            return None
        try:
            faces = cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
            )
        except Exception:
            return None
        if len(faces) == 0:
            return None
        # Return the largest face by area
        areas = [w * h for (x, y, w, h) in faces]
        return tuple(faces[np.argmax(areas)])  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Main analysis entry point
    # ------------------------------------------------------------------

    def analyze(self, video_path: str) -> Dict[str, Any]:
        """
        Performs full forensic analysis of a video file.

        Args:
            video_path: Absolute path to a readable MP4/MOV/WebM file.

        Returns:
            Dict matching the VideoForensicsDetails schema fields plus extras.
        """
        start_time = time.time()
        cap = cv2.VideoCapture(video_path)

        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        frame_indices = self._select_frame_indices(total_frames, source_fps)

        if not frame_indices:
            cap.release()
            return self._empty_result(total_frames, source_fps)

        # --- Per-frame analysis ---
        localizer = self._get_localizer()
        synth_detector = self._get_synthetic_detector()

        per_frame_manip: List[float] = []
        per_frame_synth: List[float] = []
        per_frame_temporal: List[float] = []
        sampled_frames: List[np.ndarray] = []
        suspicious_frames_data: List[Dict[str, Any]] = []

        face_detected_in_any_frame = False
        prev_gray: Optional[np.ndarray] = None
        prev_face_roi: Optional[Tuple[int, int, int, int]] = None

        worst_score = 0.0
        worst_frame_heatmap: Optional[str] = None
        worst_frame_thumb: Optional[str] = None
        worst_frame_ts: Optional[float] = None

        frame_idx_set = set(frame_indices)

        for idx in frame_indices:
            # Seek to target frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                continue
            sampled_frames.append(frame_bgr.copy())

            timestamp_sec = round(idx / source_fps, 3) if source_fps > 0 else 0.0

            # Convert to PIL for image engines
            pil_img = _frame_to_pil(frame_bgr)
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            # --- Face detection ---
            face_roi = self._detect_largest_face(gray)
            if face_roi is not None:
                face_detected_in_any_frame = True

            # --- Temporal inconsistency (requires previous frame) ---
            temporal_score = 0.0
            if prev_gray is not None:
                temporal_score = _compute_optical_flow_inconsistency(
                    prev_gray, gray,
                    face_roi=face_roi or prev_face_roi,
                )
                per_frame_temporal.append(temporal_score)

            # --- Image forensics (reusing existing engines) ---
            try:
                manip_score, _mask, heatmap_b64 = localizer.generate_heatmap(pil_img)
            except Exception:
                manip_score, heatmap_b64 = 0.0, None

            try:
                synth_result = synth_detector.analyze(pil_img, deep_scan=False)
                synth_score = float(synth_result["synthetic_score"])
            except Exception:
                synth_score = 0.0

            per_frame_manip.append(manip_score)
            per_frame_synth.append(synth_score)

            # --- Suspicious frame detection ---
            # A frame is suspicious if either manipulation or synthetic score
            # exceeds the threshold, or temporal inconsistency is very high.
            frame_overall = max(manip_score, synth_score, temporal_score)
            if face_roi is not None:
                # Boost score slightly when a face is present and score is elevated
                frame_overall = max(frame_overall, 0.4 * temporal_score + 0.6 * frame_overall)

            is_suspicious = frame_overall >= SUSPICIOUS_FRAME_THRESHOLD

            # Always record the worst frame
            if frame_overall > worst_score:
                worst_score = frame_overall
                worst_frame_heatmap = heatmap_b64
                worst_frame_thumb = _encode_pil_thumbnail(pil_img)
                worst_frame_ts = timestamp_sec

            if is_suspicious:
                thumb_b64 = _encode_pil_thumbnail(pil_img)
                suspicious_frames_data.append({
                    "frame_index": idx,
                    "timestamp_sec": timestamp_sec,
                    "manipulation_score": round(manip_score, 4),
                    "synthetic_score": round(synth_score, 4),
                    "temporal_inconsistency_score": round(temporal_score, 4),
                    "face_detected": face_roi is not None,
                    "heatmap_base64": heatmap_b64,
                    "thumbnail_base64": thumb_b64,
                })

            prev_gray = gray
            prev_face_roi = face_roi

        # VideoMAE must receive a complete evenly distributed temporal clip,
        # independent of the slower 1-FPS frame-forensics sampling above.
        if not sampled_frames:
            pass
        else:
            model_indices = np.linspace(
                0, max(0, total_frames - 1), VIDEOMAE_FRAMES
            ).round().astype(int)
            model_cap = cv2.VideoCapture(video_path)
            model_frames: List[np.ndarray] = []
            try:
                for idx in model_indices:
                    model_cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                    ret, frame_bgr = model_cap.read()
                    if ret and frame_bgr is not None:
                        model_frames.append(frame_bgr.copy())
            finally:
                model_cap.release()
            if len(model_frames) == VIDEOMAE_FRAMES:
                sampled_frames = model_frames

        cap.release()

        # --- Aggregate scores ---
        mean_manip = float(np.mean(per_frame_manip)) if per_frame_manip else 0.0
        mean_synth = float(np.mean(per_frame_synth)) if per_frame_synth else 0.0
        mean_temporal = float(np.mean(per_frame_temporal)) if per_frame_temporal else 0.0
        max_manip = float(np.max(per_frame_manip)) if per_frame_manip else 0.0
        max_synth = float(np.max(per_frame_synth)) if per_frame_synth else 0.0
        max_temporal = float(np.max(per_frame_temporal)) if per_frame_temporal else 0.0
        video_model_score = self._analyze_videomae(sampled_frames)

        # --- Deepfake score fusion ---
        # Weight: mean-frame synthetic (40%) + max-frame synthetic (20%) +
        #         temporal inconsistency (25%) + manipulation signal (15%)
        heuristic_deepfake_score = float(np.clip(
            0.40 * mean_synth
            + 0.20 * max_synth
            + 0.25 * max_temporal
            + 0.15 * max_manip,
            0.0, 1.0
        ))
        # A fine-tuned VideoMAE score is the primary score when available.
        # The heuristic remains available for explanations and fallback only.
        deepfake_score = (
            video_model_score
            if video_model_score is not None
            else heuristic_deepfake_score
        )

        # Boost deepfake score when faces are present and score is elevated —
        # facial region deepfakes produce stronger temporal artefacts.
        if (
            video_model_score is None
            and face_detected_in_any_frame
            and deepfake_score > 0.3
        ):
            deepfake_score = float(np.clip(deepfake_score * 1.15, 0.0, 1.0))

        temporal_inconsistency_score = float(np.clip(
            0.5 * mean_temporal + 0.5 * max_temporal, 0.0, 1.0
        ))

        elapsed_ms = (time.time() - start_time) * 1000.0

        return {
            "mean_manipulation_score": round(mean_manip, 4),
            "mean_synthetic_score": round(mean_synth, 4),
            "deepfake_score": round(deepfake_score, 4),
            "video_model_score": (
                round(video_model_score, 4) if video_model_score is not None else None
            ),
            "video_model_status": self._videomae_status,
            "video_model_kind": self._videomae_kind,
            "video_model_threshold": round(self._video_model_threshold, 6),
            "temporal_inconsistency_score": round(temporal_inconsistency_score, 4),
            "face_detected_in_video": face_detected_in_any_frame,
            "frame_count_total": total_frames,
            "frames_analyzed": len(per_frame_manip),
            "sample_fps": self.sample_fps,
            "suspicious_frames": suspicious_frames_data,
            "worst_frame_heatmap_base64": worst_frame_heatmap,
            "worst_frame_thumbnail_base64": worst_frame_thumb,
            "worst_frame_timestamp_sec": worst_frame_ts,
            "latency_ms": round(elapsed_ms, 1),
            # Extra diagnostic fields (not in schema, used for debugging)
            "max_manipulation_score": round(max_manip, 4),
            "max_synthetic_score": round(max_synth, 4),
            "max_temporal_score": round(max_temporal, 4),
        }

    def _empty_result(self, total_frames: int, source_fps: float) -> Dict[str, Any]:
        """Returns a safe zero-score result when the video cannot be sampled."""
        return {
            "mean_manipulation_score": 0.0,
            "mean_synthetic_score": 0.0,
            "deepfake_score": 0.0,
            "video_model_score": None,
            "video_model_status": self._videomae_status,
            "video_model_kind": self._videomae_kind,
            "video_model_threshold": round(self._video_model_threshold, 6),
            "temporal_inconsistency_score": 0.0,
            "face_detected_in_video": False,
            "frame_count_total": total_frames,
            "frames_analyzed": 0,
            "sample_fps": self.sample_fps,
            "suspicious_frames": [],
            "worst_frame_heatmap_base64": None,
            "worst_frame_thumbnail_base64": None,
            "worst_frame_timestamp_sec": None,
            "latency_ms": 0.0,
            "max_manipulation_score": 0.0,
            "max_synthetic_score": 0.0,
            "max_temporal_score": 0.0,
        }
