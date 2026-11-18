# NewsProof — Image Authenticity Verification for Online News

**v1.0 — Image-Only Vertical Slice**

NewsProof is an open, end-to-end media authenticity verification system that combines **cryptographic provenance (C2PA)** with a **calibrated multi-signal AI forensics ensemble** to deliver explainable, uncertainty-aware verdicts on news photographs.

---

## Key Capabilities

| Feature | Status |
|---|---|
| C2PA manifest parsing & trust verification | ✅ |
| SHA-256 + perceptual hash (pHash/dHash) content fingerprinting | ✅ |
| Durable credential registry (pHash-based recovery of stripped metadata) | ✅ |
| SRM noise-residual manipulation localization with RGBA heatmap | ✅ |
| 2D FFT spectral fingerprint AI-generated image detection | ✅ |
| ELA + double-JPEG DCT compression forensics | ✅ |
| Auxiliary PRNU sensor noise inconsistency | ✅ (auxiliary-only) |
| Isotonic calibration with dual ECE reporting | ✅ |
| Monte Carlo Dropout epistemic uncertainty routing | ✅ |
| 8-verdict deterministic fusion engine + core invariant | ✅ |
| DIRE deep-scan mode (optional, off by default) | ✅ (stub) |
| Append-only Merkle transparency ledger with inclusion proofs | ✅ |
| PDF audit certificate + Schema.org JSON-LD report | ✅ |
| Interactive React/HTML dashboard with heatmap overlay slider | ✅ |
| Public verify page (`/v/{asset_id}`) | ✅ |
| REST API (`/api/v1/verify`, `/manifest`, `/report`, `/ledger`) | ✅ |
| Docker Compose (full stack local dev) | ✅ |
| Video pipeline (VideoMAE + frame/temporal heuristics) | ⚠️ Experimental |
| Browser extension (Chrome Manifest V3) | 🔜 v2 |
| CMS / WordPress plugin | 🔜 v2 |
| Kubernetes + horizontal GPU autoscaling | 🔜 v2 |

---

## The 8 Verdicts

| Verdict | Trigger Condition |
|---|---|
| **Verified Authentic** | Valid C2PA signature + trusted root + hash intact + clean forensics |
| **Authentic – Edited (Declared)** | Valid C2PA + declared editorial actions match forensic fingerprint |
| **Likely Authentic** | No C2PA, but all forensic signals below clean threshold |
| **Suspicious / Manipulated** | Elevated SRM tamper score (> 0.55); no valid C2PA |
| **AI-Generated** | Elevated spectral / spatial synthetic score (> 0.55) |
| **Provenance Broken** | C2PA manifest present but hash binding or signature verification failed |
| **Unverifiable** | Below 256px, heavy compression, or MC-Dropout variance > 0.065 |
| **Discrepancy: Requires Editorial Review** | ⚠️ **Core Invariant**: Valid C2PA coexists with forensic anomaly → route to human |

> **Core Invariant**: A valid C2PA cryptographic signature is never silently overridden by an isolated low-confidence AI score. When a valid credential and a forensic flag coexist, the system returns `Discrepancy: Requires Editorial Review` and displays both the manifest and the localization heatmap.

---

## Quick Start (Local Dev)

### Prerequisites
- Python 3.11+, pip
- Docker Desktop (for PostgreSQL, Redis, MinIO via Compose)

### Without Docker (pure Python)

```bash
git clone <repo>
cd "ISC Project"
pip install -r requirements.txt

# Run API server
uvicorn apps.api.app.main:app --host 0.0.0.0 --port 8000 --reload

# Open browser
# http://localhost:8000  → Full dashboard
# http://localhost:8000/v/{asset_id}  → Public verify page
```

### With Docker Compose (full stack)

```bash
docker compose up --build
# API: http://localhost:8000
# MinIO Console: http://localhost:9001
```

### Run All Tests

```bash
python -m pytest tests/ -v
```

Run the test suite to validate the current implementation; results may grow as
new security and evaluation tests are added.

### Run Benchmark Evaluation

```bash
python data/benchmarks/run_eval.py
# Outputs: docs/EVAL.md with measured metrics
```

### Run Robustness and Dataset Checks

```bash
# Train the spatial AI detector on Tiny-GenImage
python data/benchmarks/tiny_genimage.py --train-per-class 4000 --validation-per-class 1000 --epochs 5

# Measure pHash/dHash stability under common image transformations
python data/benchmarks/fingerprint_robustness.py

# Validate a downloaded dataset manifest for cross-split content leakage
python -c "from data.benchmarks.dataset_evaluation import load_manifest, validate_manifest; from pathlib import Path; print(validate_manifest(load_manifest(Path('path/to/manifest.csv'))))"

# Evaluate video scores after re-encoding, resizing, trimming, and frame dropping
python data/benchmarks/video_robustness.py --video path/to/video.mp4

# Fine-tune VideoMAE on the downloaded Celeb-DF dataset
python data/benchmarks/train_videomae.py --dataset . --per-class 100 --epochs 5 --batch-size 1
# Optional override:
# PowerShell: $env:NEWSPROOF_VIDEOMAE_WEIGHTS = "models/celebdf_videomae_100/videomae-celebdf"
```

Video analysis uses 16 VideoMAE frames, face-aware crop sampling, motion-compensated
temporal heuristics, and existing frame-level image forensics. The 100-video
Celeb-DF run used 80 real/80 fake training clips and 20 real/20 fake source-disjoint
validation clips. Held-out accuracy was 67.5% and ROC-AUC was 0.63, so the pipeline
remains experimental and must not be presented as production-grade.
`NEWSPROOF_VIDEOMAE_WEIGHTS` and `NEWSPROOF_VIDEO_MODEL` can override the checkpoint.

The security assumptions, attack cases, and remaining deployment gaps are
documented in [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md).

The Tiny-GenImage training command writes local weights to
`models/tiny_genimage_spatial.pt`. The API automatically loads that artifact
when present. The checked-in architecture does not include binary weights;
run the training command after cloning the repository.

---

## Project Structure

```
ISC Project/
├── apps/
│   ├── api/app/
│   │   ├── main.py                    # FastAPI server: /verify, /manifest, /report, /ledger
│   │   └── schemas/verdict.py         # Pydantic v2 schemas for all 8 verdicts
│   ├── web/
│   │   ├── index.html                 # Dashboard (Tailwind CSS, heatmap slider, signals)
│   │   └── public_verify.html         # Public /v/{id} verification page
│   └── extension/README.md            # Chrome extension stub (v2)
├── engines/
│   ├── crypto/
│   │   ├── hasher.py                  # SHA-256 + pHash + dHash + Hamming distance
│   │   ├── c2pa_validator.py          # C2PA manifest parser & trust verification
│   │   ├── trust_store.py             # Trusted roots + revocation list
│   │   ├── signer.py                  # Publisher signing service (test fixtures)
│   │   └── durable_registry.py        # pHash-based credential recovery registry
│   ├── forensics/
│   │   ├── manipulation/localizer.py  # SRM noise residual + heatmap generator
│   │   ├── synthetic/detector.py      # FFT spectral + CNN spatial + MC Dropout
│   │   ├── noise_compression/         # ELA, double-JPEG DCT, PRNU auxiliary
│   │   ├── calibrator.py              # Isotonic regression + dual ECE tracking
│   │   └── deepfake_video/__init__.py # VideoMAE + temporal/frame heuristics
│   └── fusion/
│       ├── rules_engine.py            # 8-verdict deterministic fusion + core invariant
│       ├── merkle_ledger.py           # Append-only SHA-256 Merkle tree
│       └── report_generator.py        # PDF + JSON-LD Schema.org report
├── data/
│   ├── benchmarks/
│   │   ├── run_eval.py                # Full benchmark evaluation runner
│   │   ├── benchmark_datasets.py      # Corpus builder + dataset source documentation
│   │   ├── dataset_evaluation.py     # Manifest leakage and localization metrics
│   │   ├── fingerprint_robustness.py # Image fingerprint transformation experiment
│   │   └── video_robustness.py       # Video editorial-transformation experiment
│   └── test_samples/certs/            # C2PA conformance fixture certificates
├── tests/                             # 29-test suite (M0–M6)
├── docs/
│   ├── PLAN.md                        # Milestone table + task list
│   ├── DECISIONS.md                   # Architectural decision log
│   ├── EVAL.md                        # Benchmark results (real measured metrics)
│   ├── MODEL_CARD.md                  # Detector descriptions + known limitations
│   └── THREAT_MODEL.md                # Security threats, controls, and gaps
├── docker-compose.yml
├── requirements.txt
└── README.md                          ← this file
```

---

## Benchmark Results (v1 — Synthetic Evaluation Corpus)

| Metric | Target | v1 Result |
|---|---|---|
| Manipulation F1 | ≥ 0.90 | **0.600** |
| AI-Generated AUC (Known Generators) | ≥ 0.95 | **0.993** |
| AI-Generated AUC (Unseen Generators) | ≥ 0.85 | **1.000** |
| False-Positive Rate (Authentic) | ≤ 3.0% | **57.50%** |
| Median Image Latency | < 5s | **0.079s** |
| p95 Image Latency | < 10s | **0.097s** |
| In-Distribution ECE | < 0.05 | **~0.000** |

> ⚠️ **Important**: The manipulation and authentic-media results currently miss their targets. These numbers are from a **controlled synthetic evaluation corpus**, not live wire-agency archives. See [`docs/EVAL.md`](docs/EVAL.md), [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md), and [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md) for limitations.

---

## Known Limitations

1. **Proprietary wire-agency corpora not used** — FPR and F1 targets validated on freely available substitutes (CASIA v2, Columbia, CoMoFoD, GenImage, Wikimedia Commons).  
2. **Spatial CNN is randomly initialized** — Track B manipulation and synthetic detection will improve significantly with fine-tuned pretrained weights (TruFor, ConvNeXt, ResNet on CASIA/GenImage).  
3. **C2PA adoption** — Most news images in the wild do not yet carry C2PA manifests. Track B carries the full verification burden in standalone mode.  
4. **PRNU degrades under heavy recompression** — Explicitly auxiliary-only (weight = 0.15). Never drives a verdict alone.  
5. **Novel generative models** — New diffusion/autoregressive architectures may require spectral branch retraining. Modular plugin architecture supports monthly fine-tuning updates.  
6. **C2PA scope** — Validated against the C2PA conformance fixtures, not a claim of full spec compliance across all edge cases.

---

## Next (v2 Roadmap)

- 🎬 **Video deepfake detection** — Face landmark jitter, temporal blinking cadence, SyncNet lip-sync consistency, dense keyframe sampling strategy (FR-20)
- 🌐 **Chrome / Chromium Extension** — Manifest V3; right-click "Verify with NewsProof"; in-page trust pill badge injection on `<img>` tags
- 📰 **CMS Plugin** — WordPress REST webhook and Drupal module for inline editorial verification inside the authoring workflow
- ☁️ **Kubernetes + GPU autoscaling** — Helm chart for production cluster with horizontal pod autoscaling on Celery GPU worker pools
- 🔁 **Active fine-tuning pipeline** — Monthly model refresh on community-contributed synthetic and spliced media corpus
- 🔗 **C2PA durable registry with PostgreSQL + pgvector** — Replace in-memory pHash lookup with HNSW approximate nearest-neighbor index

