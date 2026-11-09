# NewsProof Implementation Plan (v1 Image Slice)

## Milestone Table

| Milestone | Scope | Deliverables |
|---|---|---|
| **M0: Scaffolding** | Repo structure, schemas, docker, test harness | Directory skeleton, Pydantic schemas, docker-compose, pytest setup |
| **M1: Track A Provenance** | Ingestion, hashing & C2PA validation | Ingestion pipeline (SHA256/pHash), C2PA parser & conformance tests, trust store |
| **M2: Manipulation & Compression** | Spatial tampering & compression forensics | Noise residual/SRM localizer, ELA, double-JPEG analysis, heatmap generator |
| **M3: Synthetic Detection** | AI generator detection & held-out eval | Spatial backbone + 2D FFT/DCT branch, deep scan stub, unseen generator test |
| **M4: Calibration & Fusion** | Calibrator, uncertainty & 8-verdict rules | Temperature/isotonic calibrator, MC uncertainty, fusion engine, all 8 verdict tests |
| **M5: Ledger & Reporting** | Transparency log, pHash registry & reports | SHA-256 Merkle tree ledger, durable pHash registry, PDF & JSON-LD report generator |
| **M6: API & UI Dashboard** | FastAPI endpoints & interactive frontend | `/verify`, `/manifest`, `/report`, `/ledger`, React dashboard with heatmap & public verify |
| **M7: Evaluation & Docs** | Benchmark runs, model cards, docs | `docs/EVAL.md` real metrics, model cards, README with limitations & Next |

## Core Task List

1. **M0**: Scaffold `apps/`, `engines/`, `data/`, `docs/`, `docker-compose.yml`, Pydantic schemas, configure pytest.
2. **M1**: Implement `engines/crypto/c2pa_validator.py`, `trust_store.py`, `signer.py`, ingestion hasher; test against C2PA fixtures.
3. **M2**: Implement `engines/forensics/manipulation/` (noise residual + Cat/TruFor-style head), `noise_compression/` (ELA + DCT), heatmap output.
4. **M3**: Implement `engines/forensics/synthetic/` (spatial + FFT/DCT spectral classifier), optional DIRE deep scan flag.
5. **M4**: Implement `engines/forensics/calibrator.py`, `engines/fusion/rules_engine.py` (8 verdicts, core invariant, high-variance fallback), test suite.
6. **M5**: Implement `engines/fusion/merkle_ledger.py`, `engines/crypto/durable_registry.py`, `engines/fusion/report_generator.py` (PDF + JSON-LD).
7. **M6**: Implement FastAPI server (`apps/api`), web dashboard and public verify page (`apps/web`), extension stub (`apps/extension`).
8. **M7**: Run benchmark evaluation (`data/benchmarks/run_eval.py`), generate `docs/EVAL.md`, write model cards, comprehensive README.

## Architectural Decisions Logged

- 8 verdicts with `Discrepancy: Requires Editorial Review` as first-class verdict.
- ConvNeXt/ResNet + FFT spectral branch default for synthetic detection; DIRE reserved behind `--deep-scan`.
- Expected Calibration Error (ECE) < 0.05 enforced for in-distribution data; held-out reported separately.
- PRNU acts strictly as auxiliary signal; high epistemic variance routes safely to `Unverifiable`.
- Tested and reported against freely available benchmark suites (CASIA v2 / Columbia / GenImage / Wikimedia).
