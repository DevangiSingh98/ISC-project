import os
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from engines.crypto.signer import C2PASigner

def create_sample_images():
    samples_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(samples_dir, exist_ok=True)
    signer = C2PASigner()

    print("Generating representative test sample images...")

    # 1. Authentic Press Photo
    np.random.seed(42)
    x = np.linspace(0, 12, 400)
    y = np.linspace(0, 12, 300)
    xx, yy = np.meshgrid(x, y)
    texture = np.sin(xx / 2.0) + np.cos(yy / 3.0)
    noise = np.random.normal(128, 12, (300, 400))
    raw = np.clip(texture * 35 + noise, 0, 255).astype(np.uint8)
    auth_img = Image.fromarray(np.dstack([raw, raw, raw]))
    auth_path = os.path.join(samples_dir, "authentic_press.jpg")
    auth_img.save(auth_path, "JPEG", quality=92)
    print(f"Created {auth_path}")

    # 2. C2PA Signed Authentic
    c2pa_auth_path = os.path.join(samples_dir, "c2pa_signed_authentic.jpg")
    signer.sign_asset(
        input_image_path=auth_path,
        output_image_path=c2pa_auth_path,
        title="Breaking News Coverage",
        publisher="Reuters News Credential Authority",
        actions=[{"action": "c2pa.created", "digitalSourceType": "http://cv.iptc.org/newscodes/digitalsourcetype/digitalCapture"}]
    )
    print(f"Created {c2pa_auth_path}")

    # 3. C2PA Signed with Declared Edits
    c2pa_edit_path = os.path.join(samples_dir, "c2pa_edited_declared.jpg")
    signer.sign_asset(
        input_image_path=auth_path,
        output_image_path=c2pa_edit_path,
        title="Capitol Hill Press Photo",
        publisher="Associated Press C2PA Root",
        actions=[
            {"action": "c2pa.created", "digitalSourceType": "http://cv.iptc.org/newscodes/digitalsourcetype/digitalCapture"},
            {"action": "c2pa.cropped"},
            {"action": "c2pa.color_adjusted"}
        ]
    )
    print(f"Created {c2pa_edit_path}")

    # 4. C2PA Broken (Tampered post-signing)
    c2pa_broken_path = os.path.join(samples_dir, "c2pa_broken_tampered.jpg")
    with open(c2pa_auth_path, "rb") as f:
        tampered_bytes = bytearray(f.read())
    # Modify 80 bytes in the compressed image stream
    tampered_bytes[1500:1580] = b"\xAA" * 80
    with open(c2pa_broken_path, "wb") as f:
        f.write(tampered_bytes)
    print(f"Created {c2pa_broken_path}")

    # 5. Spliced / Manipulated Image
    manip_img = auth_img.copy()
    np.random.seed(99)
    foreign_patch = np.zeros((100, 100, 3), dtype=np.uint8)
    for c in range(3):
        foreign_patch[:, :, c] = np.random.normal(210, 45, (100, 100)).clip(0, 255).astype(np.uint8)
    foreign_pil = Image.fromarray(foreign_patch)
    manip_img.paste(foreign_pil, (150, 90))
    spliced_path = os.path.join(samples_dir, "spliced_manipulated.jpg")
    manip_img.save(spliced_path, "JPEG", quality=85)
    print(f"Created {spliced_path}")

    # 6. C2PA Discrepancy (Core Invariant test sample)
    c2pa_discrepancy_path = os.path.join(samples_dir, "c2pa_discrepancy.jpg")
    signer.sign_asset(
        input_image_path=spliced_path,
        output_image_path=c2pa_discrepancy_path,
        title="Investigative Report Visual",
        publisher="NewsProof Official Trust Root",
        actions=[{"action": "c2pa.created", "digitalSourceType": "http://cv.iptc.org/newscodes/digitalsourcetype/digitalCapture"}]
    )
    print(f"Created {c2pa_discrepancy_path}")

    # 7. AI Synthetic Diffusion Image
    np.random.seed(202)
    y, x = np.ogrid[:300, :400]
    checker = np.sin(x * np.pi / 2.0) * np.sin(y * np.pi / 2.0)
    synth_noise = np.random.normal(128, 10, (300, 400))
    synth_arr = np.clip(synth_noise + checker * 45.0, 0, 255).astype(np.uint8)
    synth_img = Image.fromarray(np.dstack([synth_arr, synth_arr, synth_arr]))
    synth_path = os.path.join(samples_dir, "ai_synthetic_diffusion.jpg")
    synth_img.save(synth_path, "JPEG", quality=90)
    print(f"Created {synth_path}")

    print("All sample media generated successfully!")

if __name__ == "__main__":
    create_sample_images()

