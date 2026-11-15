import hashlib
from typing import Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image, ExifTags
from scipy.fftpack import dct

def compute_sha256(data: bytes) -> str:
    """Computes standard SHA-256 hex digest of file bytes."""
    return hashlib.sha256(data).hexdigest()

def compute_phash(image: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> str:
    """
    Computes 64-bit Discrete Cosine Transform (DCT) Perceptual Hash (pHash).
    Robust to minor resizing and compression.
    """
    img_size = hash_size * highfreq_factor
    # Convert to grayscale and resize smoothly
    gray_img = image.convert("L").resize((img_size, img_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray_img, dtype=np.float32)
    
    # 2D DCT
    dct_rows = dct(pixels, axis=0, norm='ortho')
    dct_2d = dct(dct_rows, axis=1, norm='ortho')
    
    # Extract top-left low frequencies (excluding DC term at [0, 0])
    dct_lowfreq = dct_2d[:hash_size, :hash_size]
    med = np.median(dct_lowfreq[1:, 1:])
    
    diff = dct_lowfreq > med
    # Convert boolean array to hex string
    bit_string = "".join(["1" if b else "0" for b in diff.flatten()])
    return f"{int(bit_string, 2):016x}"

def compute_dhash(image: Image.Image, hash_size: int = 8) -> str:
    """Difference Hash (dHash) tracking horizontal gradient changes."""
    gray = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    arr = np.asarray(gray, dtype=np.int32)
    diff = arr[:, 1:] > arr[:, :-1]
    bit_string = "".join(["1" if b else "0" for b in diff.flatten()])
    return f"{int(bit_string, 2):016x}"

def hamming_distance(hex_hash1: str, hex_hash2: str) -> int:
    """Computes Hamming distance between two 16-hex-digit hashes."""
    val1 = int(hex_hash1, 16)
    val2 = int(hex_hash2, 16)
    xor = val1 ^ val2
    return bin(xor).count("1")

def inspect_video_metadata(video_path: str) -> Dict[str, Any]:
    """
    Extracts container-level metadata from a video file using cv2.VideoCapture.
    Returns duration, fps, frame_count, resolution, and codec.
    Safe to call before frame extraction; does not retain the capture object.
    """
    import cv2 as _cv2
    cap = _cv2.VideoCapture(video_path)
    try:
        fps = float(cap.get(_cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(_cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(_cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fourcc_int = int(cap.get(_cv2.CAP_PROP_FOURCC))
        # Decode FourCC integer to 4-character string
        codec = "".join([chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)]).strip("\x00")
        duration_sec = (frame_count / fps) if fps > 0 else 0.0
    finally:
        cap.release()

    return {
        "duration_sec": round(duration_sec, 3),
        "fps": round(fps, 3),
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "codec": codec or "UNKNOWN",
    }


def inspect_image_metadata(image: Image.Image, raw_bytes: bytes) -> Dict[str, Any]:
    """Extracts format, dimensions, EXIF data, and assesses degradation."""
    width, height = image.size
    is_degraded = (width < 256 or height < 256)
    
    exif_data: Dict[str, Any] = {}
    try:
        raw_exif = image.getexif()
        if raw_exif:
            for tag_id, value in raw_exif.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                # Convert bytes to string representation if needed
                if isinstance(value, bytes):
                    value = value.decode("utf-8", errors="ignore")
                exif_data[tag_name] = str(value)
    except Exception:
        pass

    return {
        "width": width,
        "height": height,
        "format": image.format or "UNKNOWN",
        "mode": image.mode,
        "is_degraded": is_degraded,
        "byte_size": len(raw_bytes),
        "exif": exif_data
    }

