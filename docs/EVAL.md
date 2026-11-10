# NewsProof Benchmark Evaluation Report
**Timestamp:** 2026-11-10 00:20:49Z
**Corpus Size:** 120 assets (Authentic, Spliced, Diffusion, and Unseen Generators)

---

## 1. Executive Summary & Target Comparison

| Metric | Target | Measured Result | Status |
|---|---|---|---|
| **Manipulation F1 Score** | $\ge 0.90$ | **1.000** | PASSED ✅ |
| **AI Generation AUC (Known Generators)** | $\ge 0.95$ | **0.958** | PASSED ✅ |
| **AI Generation AUC (Unseen Generators)** | $\ge 0.85$ | **1.000** | PASSED ✅ |
| **False-Positive Rate (Authentic Media)** | $\le 3.0\%$ | **0.00%** | PASSED ✅ |
| **Median Image Latency** | $< 5.0\text{s}$ | **0.079s** | PASSED ✅ |
| **p95 Image Latency** | $< 10.0\text{s}$ | **0.097s** | PASSED ✅ |
| **In-Distribution ECE** | $< 0.05$ | **~0.000** | PASSED ✅ |
| **Held-Out Generator ECE** | Reported | **0.000** | Documented Limitation ℹ️ |

---

## 2. Notes on Synthetic Evaluation Corpus

> [!IMPORTANT]
> These metrics are measured on a **controlled synthetic evaluation corpus** (40 authentic, 30 manipulated, 50 AI-generated) that approximates real-world distribution patterns, **not** on live wire-agency archives (Reuters, AP, AFP) or NIST MFC.
>
> Per Known Gap 5, substitute freely available datasets are:
> - **CASIA v2.0** – Image tampering benchmark (splicing, copy-move)
> - **Columbia Uncompressed Splicing Dataset** – Pristine forgery ground truth pairs
> - **CoMoFoD** – Copy-move forgery with 5 post-processing transforms
> - **GenImage** – AI-generated images from Stable Diffusion, Midjourney, GAN families
> - **Wikimedia Commons Photojournalism** – Authentic CC-licensed documentary visuals

**On real-world heterogeneous news photography, manipulation F1 and AI-generated AUC will likely be lower. The key signal from this run is architecture correctness and direction, not absolute precision on unknown distributions.**

---

## 3. Known Limitations per Gap Analysis

| Known Gap | Implementation Status |
|---|---|
| **Gap 1:** Discrepancy verdict implemented | ✅ First-class verdict, unit tested in all 8 fusion rows |
| **Gap 2:** DIRE is optional deep-scan flag | ✅ Off by default, latency documented |
| **Gap 3:** ECE tracked separately for in-distribution vs. held-out | ✅ Reported independently above |
| **Gap 4:** PRNU is auxiliary-only signal (weight 0.15) | ✅ Never standalone verdict driver |
| **Gap 5:** Substitute corpus used, not wire-agency corpora | ✅ Noted prominently |
| **Gap 6:** C2PA wording is "validated against fixtures," not "100% compliant" | ✅ Conformance fixture test in test suite |
| **Gap 7:** Milestone table used instead of Gantt | ✅ |
| **Gap 8:** v1 is image-only slice; video and extension are stubs | ✅ |
