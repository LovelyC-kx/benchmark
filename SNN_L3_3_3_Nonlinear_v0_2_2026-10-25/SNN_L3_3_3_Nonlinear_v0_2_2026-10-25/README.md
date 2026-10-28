# L3.3.3 Nonlinear interactions, conditional coding and readout error

2026-10-25, v0.2 review. Independently implemented analysis of one frozen checkpoint.
Read the accompanying Chinese protocol for definitions, literature provenance,
experimental decisions, interpretation boundaries and manuscript placement.

## Run

Install requirements-reference.txt for NumPy analyses; requirements-runtime.txt
for the SNN and trainable readout probes. Figure generation additionally needs
requirements-figures.txt. Commands below work as single PowerShell lines.
Every output file/directory passed to a runner must not already exist.

```powershell
python -m pip install -r requirements-runtime.txt
python -m unittest discover -s tests -v
python run_calibrations.py --output outputs/known_answers.json --with-mlp
python run_monte_carlo_reference.py --output outputs/mc_reference.json
python run_additivity_reference.py evidence/source_smoke_run --output outputs/additivity_reference.json
python run_nonlinear.py evidence/source_smoke_run --output outputs/c_nonlinear --all-schedules --focal C --nested --smoke --device cpu
python run_nonlinear.py evidence/source_smoke_run --output outputs/h1_nonlinear --schedule-index 0 --focal H1 --smoke --device cpu
python run_nonlinear.py evidence/source_smoke_run --output outputs/joint_nonlinear --schedule-index 0 --observation joint_post --smoke --device cpu
```

Without --smoke the analysis uses larger independent samples and all three
registered MLP initialization seeds. This DOES NOT train or improve the input
SNN checkpoint. The bundled checkpoint originally received one optimizer
update; all its present results are pipeline checks, not model findings.

## Main files

- functional_anova.py: complete group-subset covariance inversion, exclusive
  error-risk budget, paired bootstrap, repeat risk, nested conditional functions.
- factor_sampling.py: F/G/U/E factor contracts, legal hybrids, split-specific
  completion RNG, inactive-group removal. Only x reaches the SNN.
- decoding.py: independently selected linear/quadratic ridge and small MLP
  probes, exported prediction functions and explicit parameter/sample budgets.
- context_audit.py: six-by-six fixed-reader transfer and shared-readout controls;
  an extra schedule one-hot is separately labeled as an analysis permission.
- task_objective.py: original anchored/uniform regression weights preserved
  in the whole-model error budget; annotate_objectives.py applies the same
  calculation to already exported frozen observations without another fit.
- run_nonlinear.py: frozen-model integration; no SNN optimizer/audit access.
- run_calibrations.py, run_monte_carlo_reference.py: known-answer and estimator
  calibration, explicitly distinct from SNN science.
- run_additivity_reference.py: actual-architecture algebraic check with the
  SAME weights and a continuous forward override; not a trained baseline.
- tests/test_nonlinear.py: 63 checked cases (53 original + 10 review cases) when PyTorch is available; one
  optional PyTorch test is skipped by the NumPy-only installation.
- parent_provenance.json: 17 byte-identical internal parent files, not imported
  paper-author implementations. Upstream author code is NOT redistributed.

## Evidence

Final eight views are in evidence/validated_v1_c_six_schedules,
evidence/validated_h1_schedule0 and evidence/validated_joint_schedule0.
All eight refer to ONE checkpoint 272ae6230942ebeb. The SNN weights do not
change. Readout training is performed and is reported separately: 104 probe
fit calls across these views/context controls, including 32 MLP fits; 5,120
MLP updates in that SNN-analysis portion. Calibration readouts are additional
and are not SNN training runs. No final audit data were accessed.

The separate additivity reference uses two data-role contracts and a copied
continuous forward override in float64. It is not part of the eight original
model views and not an independently trained SNN baseline.

All 2^m combinations are complete ONLY for the active declared GROUPS. The
method does not exhaustively resolve all coordinates within F/G/U or every
measurement-noise entry. Sobol variance is not mutual information; dependence
is not recoverability; error attribution is not causal deletion benefit.

## v0.2 review

The runner now appends symmetric-difference ANOVA and a paired A/B risk
budget automatically. Existing legacy results are retained. The symmetric
estimator has exact additive interaction zeros but is not uniformly
lower-variance for nonzero main effects.

`python run_estimator_review.py --output outputs/estimator_review.json`

`python robust_review.py evidence/validated_v1_c_six_schedules --draws 64`

Stored eight-view updates require no new SNN or probe fits. A separate fresh
schedule-0 end-to-end rerun is in evidence/validated_v0_2_fresh_schedule0.
The six-map explicit context gate changes slopes; linear features plus a
schedule one-hot alone change only intercepts.
