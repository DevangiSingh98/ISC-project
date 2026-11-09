# Architecture Decisions Log

- 2026-11-10: v1 scope fixed to image-only vertical slice; video and browser extension preserved as clean interfaces/stubs.
- 2026-11-10: Implemented 8 first-class verdicts including 'Discrepancy: Requires Editorial Review'.
- 2026-11-10: Default synthetic detector is spatial ConvNeXt/ResNet + 2D FFT/DCT power spectrum branch; DIRE placed behind optional deep-scan flag.
- 2026-11-10: Temperature scaling and isotonic regression calibrate in-distribution predictions (target ECE < 0.05); out-of-distribution reported separately.
- 2026-11-10: PRNU and ELA treated as auxiliary supporting signals; never standalone verdict drivers under heavy compression.
- 2026-11-10: High model uncertainty or variance routes safely to 'Unverifiable' rather than false alarms.
- 2026-11-10: Evaluation benchmarks utilize freely accessible datasets (CASIA v2, Columbia, CoMoFoD, GenImage, Wikimedia Commons) with substitute corpus note.
- 2026-11-10: C2PA conformance documented as "validated against the C2PA conformance fixtures" with UI warning on untrusted/self-signed certs.
