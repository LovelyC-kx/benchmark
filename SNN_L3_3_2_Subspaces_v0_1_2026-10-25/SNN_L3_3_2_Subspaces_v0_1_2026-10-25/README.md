# L3.3.2 coding subspaces and readout consequences

Start with `SNN_L3_3_2_Subspaces_v0_1_2026-10-25.md`. The document covers definitions, derivations, sampling, literature reuse, code, manuscript organization and the exact completion boundary.

This package analyzes one fixed checkpoint. It implements independent role-encoding fits, validation-selected ranks including zero, independent replica comparisons, conditional covariance metrics, explicit random/variance controls and readout evaluation under observation restrictions.

PowerShell-compatible single-line commands:

```powershell
python -m pip install -r requirements-reference.txt
python -m unittest discover -s tests -v
python run_calibrations.py --output outputs/known_answers.json
python run_linear_reference.py --output outputs/linear_reference.json
python -m pip install -r requirements-runtime.txt
python run_geometry.py evidence/source_smoke_run --output outputs/c_geometry --all-schedules --smoke --device cpu
```

Optional figure: install `requirements-figures.txt` and pass `--figure outputs/known_answers.png` to `run_calibrations.py`. Use a new output path each time.

The included checkpoint is the previous tiny regression checkpoint with one original optimizer update. Eight verified analysis views are eight views of this one checkpoint. No SNN optimizer is run, and no audit split is used.

Default independent natural splits use 1024 fit / 512 selection / 1024 replica / 1024 remainder-covariance / 1024 probe-fit / 512 probe-selection / 1024 evaluation scenes. Conditional panels use 256x8 fit, 128x8 selection and 256x8 evaluation. Smoke uses smaller recorded sizes. Six schedules share their latent and measurement draws to permit paired context descriptions.

Initial adapter: full-presence continuous-coordinate regression, terminal `last_post` or `joint_post`, focal `C` or `H1`. Classification and image reconstruction need task-specific contracts. No complete nonlinear information decomposition, Wakhloo four-statistic reproduction, recurrent-core causal confirmation or mature SNN ranking is claimed.

The same-rank/variance constructed control is explicitly not a uniform conditional Haar null. A failed nearest-random variance match remains marked failed. Zero selected rank produces undefined angle comparisons. Bootstrap intervals are within-checkpoint scene summaries conditional on selected ranks, not training-seed guarantees.

Verified runtime: Python 3.12.14, NumPy 2.3.5, PyTorch 2.14.1+cpu. CUDA was not used. Parent project sources are byte-identical and hash-pinned. Upstream author source is not redistributed or claimed as executed.
