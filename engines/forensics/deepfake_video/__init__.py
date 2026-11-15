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
SUSPICIOUS_FRAME_THRESHOLD = 0.40  # Score above which a frame is "suspicious"
THUMBNAIL_SIZE = (160, 90)       # Width × height for strip thumbnails


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

    # Flow magnitudes
    flow_vecs = good_next - good_prev
    magnitudes = np.linalg.norm(flow_vecs, axis=2).ravel()

    # Inconsistency = coefficient of variation of flow magnitudes
    # Natural motion produces smooth, consistent flow; deepfake artefacts
    # produce erratic per-point flow divergence.
    mean_mag = float(np.mean(magnitudes)) + 1e-6
    std_mag = float(np.std(magnitudes))
    cv_score = std_mag / mean_mag  # coefficient of variation

    # Normalise: CoV > 2.0 is considered very inconsistent
    return float(np.clip(cv_score / 2.0, 0.0, 1.0))


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
        self.version = "v2.0-full"

        # Lazy-imported image engines (avoid heavy torch init at module import time)
        self._localizer = None
        self._synthetic_detector = None
        self._face_cascade = None

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
            # Evenly subsample from already-selected indices
            subsample_step = len(indices) // MAX_FRAMES_TO_ANALYSE
            indices = indices[::subsample_step][:MAX_FRAMES_TO_ANALYSE]

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
            frame_overall = max(manip_score, synth_score)
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
                    "face_detected": face_roi is not None,
                    "heatmap_base64": heatmap_b64,
                    "thumbnail_base64": thumb_b64,
                })

            prev_gray = gray
            prev_face_roi = face_roi

        cap.release()

        # --- Aggregate scores ---
        mean_manip = float(np.mean(per_frame_manip)) if per_frame_manip else 0.0
        mean_synth = float(np.mean(per_frame_synth)) if per_frame_synth else 0.0
        mean_temporal = float(np.mean(per_frame_temporal)) if per_frame_temporal else 0.0
        max_manip = float(np.max(per_frame_manip)) if per_frame_manip else 0.0
        max_synth = float(np.max(per_frame_synth)) if per_frame_synth else 0.0
        max_temporal = float(np.max(per_frame_temporal)) if per_frame_temporal else 0.0

        # --- Deepfake score fusion ---
        # Weight: mean-frame synthetic (40%) + max-frame synthetic (20%) +
        #         temporal inconsistency (25%) + manipulation signal (15%)
        deepfake_score = float(np.clip(
            0.40 * mean_synth
            + 0.20 * max_synth
            + 0.25 * max_temporal
            + 0.15 * max_manip,
            0.0, 1.0
        ))

        # Boost deepfake score when faces are present and score is elevated —
        # facial region deepfakes produce stronger temporal artefacts.
        if face_detected_in_any_frame and deepfake_score > 0.3:
            deepfake_score = float(np.clip(deepfake_score * 1.15, 0.0, 1.0))

        temporal_inconsistency_score = float(np.clip(
            0.5 * mean_temporal + 0.5 * max_temporal, 0.0, 1.0
        ))

        elapsed_ms = (time.time() - start_time) * 1000.0

        return {
            "mean_manipulation_score": round(mean_manip, 4),
            "mean_synthetic_score": round(mean_synth, 4),
            "deepfake_score": round(deepfake_score, 4),
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
