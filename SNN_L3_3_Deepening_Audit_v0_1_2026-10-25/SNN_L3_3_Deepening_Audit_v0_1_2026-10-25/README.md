# L3.3 deepening audit v0.1

Read `SNN_L3_3_Deepening_Audit_v0_1_2026-10-25.md` first. It distinguishes new analysis, parent-code reuse, classic theory, completed checks and pending science.

This package implements declared-readout risk accounting, conditional repeated risk, Gaussian covariance controls, role-based pick-freeze sensitivity and discrete event-path diagnostics. It includes one existing tiny L3.2 checkpoint for portable pipeline verification, not a new benchmark model.

PowerShell-compatible single-line commands, run from this directory:

```powershell
python -m pip install -r requirements-reference.txt
python -m unittest discover -s tests -v
python run_calibrations.py --output outputs/calibrations.json
python -m pip install -r requirements-runtime.txt
python run_snn_deepening.py evidence/source_smoke_run --output outputs/snn_smoke --smoke --device cpu
```

Use a new output path for each run. No analysis script accesses audit_test or updates an SNN optimizer. Diagnostic encoders/probes are fitted on their own declared development draws.

Runtime verified: Python 3.12.14, NumPy 2.3.5, PyTorch 2.14.1+cpu. No CUDA validation. The lower runtime requirement is not a claim that every supported PyTorch release has been tested.

The initial adapter is for the existing regression/full-presence/two-layer MLP/terminal-last-post contract. Classification needs a task-direction contract; reconstruction needs legal scene interventions. Subspace estimation, full functional ANOVA, causal mediation and Wakhloo theory execution are not completed by this package.

`parent_provenance.json` records byte-identical parent files. No upstream paper's author implementation is redistributed or claimed as reproduced. All evidence in `evidence/` is calibration or tiny-checkpoint workflow evidence.
