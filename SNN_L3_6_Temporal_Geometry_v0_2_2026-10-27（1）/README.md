# SNN L3.6: temporal geometry, content identity and event-sensitive analysis

Version 0.2, 2026-10-27. Start with **SNN_L3_6_Delivery_Guide_2026-10-27.md**, then **SNN_L3_6_Deep_Review_and_Execution_2026-10-26.md**. It follows seven subparts, identifies primary papers and selected author code, derives the reference laws, describes adaptations, reports actual evidence, and explains the paper's contribution and remaining experiments.

本节以同一内容的跨时访问为主线。区分同刻恢复、固定读者失配、受限对齐、来源/年龄错位，以及SNN离散事件造成的局部几何问题。正文为中文，附英文摘要；本README提供英文运行和范围说明。

## Actual scope

- **67 tests pass**, no skips. One test covers temporal export for 12 registered cells, including auxiliary states and explicit analog output. In addition, 96 cell/backbone classification cases and 24 native image reconstruction cases preserve predictions exactly at the same minibatch size.
- ONE old checkpoint, originally ONE optimizer update, unchanged SHA256 `f46effd40083643fc4d298b7390657488dd91b4fcb989c2fd13a1f3b35a8410e`.
- 9 carrier views, 5 content sources, all 16x16 source-time/target-time pairs and 4 primary transport families. **46,080 risk cells per objective**, classification and continuous recovery. These cells are correlated, not independent SNN experiments.
- 378 closed-form matrix solves, corresponding to 3,060 logical content/objective/reader fits. Two shared 128-scene draws use 222 unique training scenes in total.
- 2,048 scenes select among six readout strategies; 8,192 fresh scenes evaluate frozen choices. 45 simultaneous paired comparisons; 22 positive lower bounds.
- Supplementary 1/3/7/11 blank-delay scan, six transport families, 1,024 scenes. All 900 existing comparisons receive simultaneous conditional bounds, with 15 positive lower bounds.
- Six paired schedules, 360 source/clock rows; 12 finite-scale content-metric groups; conditional noise panel 64 scenes x 8 repeats.
- A separate conditional-mean geometry extension uses 8 contents x 128 repeated noise trials, 8 groups and 32 forward captures. This repeated-trial observer has more access than a single-trial state.
- **No new SNN optimizer updates; formal benchmark 0/435; audit_test untouched; no real dataset downloads.** CPU only; user A40 not used.

The early-time readout improvement does not extend memory at the deadline. The late affine adaptation uses extra unlabelled paired calibration and map parameters. All model evidence is from a two-layer hard-LIF MLP; 12-cell coverage is an export/interface check, not trained performance coverage.

## Reproduce

Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0 and Torch 2.14.1+cpu were used for the recorded run. Compatible existing GPU Torch installations can run the code, but bitwise reproduction is validated only in the recorded environment. Rank-deficient SVD completions can vary with numerical libraries.

~~~bash
python -m pip install -r requirements-runtime.txt
python -m unittest discover -s tests -v
python run_temporal_controls.py --output outputs/controls.json
python run_temporal_review.py --output outputs/temporal
python run_temporal_extensions.py outputs/temporal --output outputs/extensions
python run_mean_geometry.py --output outputs/mean_geometry.json
python certify_delay_readers.py --primary outputs/temporal --extensions outputs/extensions --output outputs/delay_ci.json
python plot_l36.py
python verify_l36_artifacts.py
~~~

Output paths must not exist. Plot/verifier defaults use bundled evidence. The full 16x16 primary CPU run took about 1,182 seconds in this implementation; this is not a hardware benchmark.

`--run-directory` on the primary, supplementary, mean-geometry, delay-certification and new closure runners accepts compatible project `best.pt`/`run.json` directories. The project loader and current layer mapping remain two-layer MLP adapters. Do not silently reuse those layer names for image backbones. The mathematical readers/maps accept general feature dimensions; 1/2/4 visible coordinates per source have explicit target contracts. Aux states are exported where present. `named_temporal_bank` supports all named cells across eight backbones, preserving original spatial shape metadata and actual transmitted output. It is not a complete runtime-state or DAG export. Presence-stratified tasks and other architectures need their own legal observation mapping.

Optional author checks require the separately retrieved pinned files, as listed in `references/selected_source_manifest.json`:

~~~bash
python audit_temporal_sources.py --source-directory YOUR_PINNED_SOURCE_DIRECTORY --output outputs/author_checks.json
~~~

The audit executes selected functions only, with explicitly noted fake estimators and CPU device adaptation. It does not import author top-level training or run their datasets.

## Interpretation

Temporal generalization failure is not a unique measure of forgetting. O(D) permits reflections; SO(D) does not. A better whole-state map can harm task reading. The SciPy orthogonal `scale` return is not a similarity multiplier. Global decoding can be informative when the ordinary local derivative of a spike-mediated representation is zero almost everywhere. Finite-scale changes and event crossings must therefore be examined alongside decoder performance.

Pathwise geometry, geometry of the noise-conditional mean, and probability-model information are different objects. A noisy threshold has a smooth conditional mean and nonzero Bernoulli Fisher information despite zero pathwise derivative almost everywhere. `run_mean_geometry.py` makes the extra observation permission explicit.

The 45 primary and 900 supplementary confidence families each have their stated 95% conditional guarantee; they are not advertised as a joint whole-study 95% result. Continuous MSE is not assigned the bounded classification certificate.

## Self-contained dependencies and recovery

`parent_l34/` includes 41 unchanged inherited Python files from the complete CRC-checked L3.4 v0.2 archive. The checkpoint matches that complete parent. Only its code dependencies are included; old parent evidence is not counted as current L3.6 evidence. The recovered old L3.5 ZIP was truncated; this package does not require that ZIP. The recovery issue is recorded in `references/project_provenance.json`; complete restoration of that old archive is not claimed.

References distinguish full-text/selected-method reading, pinned code, actual function execution and unresolved official code locations. Missing source packages are not replaced by guessed repositories.

## v0.2 closure and one-command reproduction

The seven-part plan and final evidence map are in the Delivery Guide. The paper hypotheses, confirmatory experiment definitions, comparison units and external data requirements are in SNN_L3_6_Paper_and_Experiment_Specification_2026-10-27.md.

- 540 offset-aware cycles and 54 equal-lag transfers; frozen maps before 2,048 new scenes.
- Six causal/side-information clock reader candidates; 25 independently selected comparisons, with zero positive simultaneous lower bounds on the old checkpoint. Known current-phase control succeeds. Phase permission and 3D+3 feature cost are explicit.
- 576 scale/threshold/noise/observer-precision rows, 585 forward captures, 128 matched scenes. The 2/4/8-bit observer quantizer has finite levels, calibration-only range and clipping; internal arithmetic is unchanged. Rare full-path crossings, quantiles, maximum and nonzero scene counts accompany means.
- Eight stages pass when replaying all supplements with the bundled frozen primary. A separate UNTRAINED continuous/q=1/20-step fixture passes all nine stages including a fresh primary and 700 delay certificates using its own hash. Neither is a trained-model replication.

~~~bash
python l36_suite.py --run-directory evidence/source_smoke_run --output outputs/l36_from_scratch
python l36_suite.py --run-directory evidence/source_smoke_run --primary-directory evidence/frozen_temporal_review --output outputs/l36_replay
python run_temporal_consistency.py evidence/frozen_temporal_review --output outputs/consistency
python run_phase_readers.py evidence/frozen_temporal_review --output outputs/phase
python run_event_scale_scan.py evidence/frozen_temporal_review --output outputs/event_scale
python run_model_interfaces.py --output outputs/interfaces.json
python plot_l36_closure.py
~~~

Use a new output path for each command. Add --print-plan to l36_suite.py to inspect its commands without running. l36_suite.py records every stage, exit code and elapsed time and stops on failure. To use a different compatible checkpoint, pass its directory consistently, including to certify_delay_readers.py and run_mean_geometry.py. Generic image model loading/training is not supplied by the old project LoadModel adapter; use named_temporal_bank on an explicitly instantiated compatible model.

Optional author source retrieval and selected function audit:

~~~bash
python retrieve_pinned_author_sources.py --output outputs/pinned_author_sources
python audit_temporal_sources.py --source-directory outputs/pinned_author_sources --output outputs/author_audit.json
~~~

The downloader verifies each exact Git blob. Original repository attribution and licensing remain with the authors; these selected checks do not run author training/data workflows. The newest verification is evidence/artifact_verification_v02.json; evidence/artifact_verification.json is the retained original v0.1 record.
