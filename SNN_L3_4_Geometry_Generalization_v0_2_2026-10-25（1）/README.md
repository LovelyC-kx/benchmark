# SNN L3.4 deep review and execution v0.2

Read **SNN_L3_4_Deep_Review_and_Execution_v0_2_2026-10-25.md** first. It preserves the original W0–W3 plan and explains literature, derivations, implementation changes, results and next formal experiments in Chinese.

This is a research protocol plus runnable analysis package. Evidence includes known-answer controls and ONE frozen checkpoint originally trained for ONE optimizer update. Formal benchmark: **0/435**. No audit-test access, no new SNN training, no user A40 use.

## Current evidence

- 55 tests pass, including the optional Torch pairing test (no skips in the validated environment).
- 13 classification semantic views × 3 role groups; shared nested p=1/8/32/128 training indices; 8192 fresh evaluation scenes per view.
- Ridge uses fixed lambda 1e-3, not a tuned performance ceiling.
- MLP runs ONLY on C final-post/sphere, F/G/FG: 7680 vectorized optimizer steps and 1536 individual task-head fits. It shares feature scaling and training indices with the linear readers.
- 13 continuous-target views; 1248 closed-form regression maps; separate 8192 evaluation scenes. Moment-calibrated scalar gain uses 2048 EXTRA labelled scenes.
- 9 higher-order estimator-control diagnostics; all selected ordinary controls in this SNN smoke checkpoint. The known nonlinear control shows a variance reduction.
- Theory-derived and directly selected spectral readouts, random-candidate mean, all negative gains, conditional scene confidence intervals and measured selection costs are preserved.

## Reproduce

Install requirements-reference.txt for numerical theory/plotting and requirements-runtime.txt for frozen SNN/readout integration. Validated with Python3.12.14, NumPy2.3.5, SciPy1.17.0, Torch2.14.1+cpu. Outputs must not already exist.

~~~bash
python -m unittest discover -s tests -v
python run_v0_2_controls.py --output outputs/known_controls.json
python run_focused_review.py evidence/source_smoke_run --output outputs/classification --smoke --all-schedules --carriers --task-priors --mlp
python run_continuous_extension.py evidence/source_smoke_run outputs/classification --output outputs/continuous
python run_higher_order_controls.py evidence/source_smoke_run outputs/classification --output outputs/higher_order_controls
python plot_v0_2.py
~~~

## Reading code

geometry_generalization → task_moments → finite_gaussian → approximation_diagnostics → task_structure → higher_order_controls → task_spectrum → continuous_risk → readout_design/paired_readouts → execution scripts.

Current reports: evidence/validated_v0_2_thirteen_views, evidence/validated_v0_2_continuous_native, evidence/validated_v0_2_higher_order_controls, evidence/known_controls_v0_2_continuous.json.

Selected author repositories are pinned by commit/blob in references/l34_v0_2_source_audit.json. Full author experiments were not executed. Author source is not bundled as if it were our code. Source-reading scope, unresolved official-code availability, and implementation adaptations are explicit in the protocol.

## History and boundaries

history/v0_1_code preserves prior root scripts. Existing v0.1 evidence and the 3.5–3.7 preliminary protocol remain for continuity; do not pool reruns as independent models. SNN_L3_4_to_3_7_Research_Protocol_2026-10-25.md now routes to this detailed 3.4 review and retains earlier 3.5–3.7 work unchanged.

Mean individual-reader risk is not deployment-ensemble risk. Continuous MSE does not use bounded classification-loss confidence intervals. Gaussian exactness refers to the stated training law with known parameters; estimated proxy parameters, numerical integration and reference-pool uncertainty remain distinct.
