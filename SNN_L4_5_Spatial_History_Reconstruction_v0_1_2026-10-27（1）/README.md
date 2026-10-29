# SNN L4.5 — Spatial history use in reconstruction

2026-10-27, v0.1. Chinese analysis, conditional mathematical results, literature/code ledger and executable development evidence. The original six layers and L4.1–L4.5 are preserved.

## Read in this order

1. `SNN_L4_5_Deep_Review_and_Execution_2026-10-27.md`: eight subsections, actual findings, paper role and design predictions.
2. `SNN_L4_5_Mathematical_Specification_2026-10-27.md`: target coverage, conditional risk, region/frequency accounting, summaries, normalization and owners.
3. `SNN_L4_5_Literature_and_Code_Map_2026-10-27.md`: four original references plus reconstruction/representation neighbors; fixed source and execution scope.
4. `SNN_L4_5_Data_Download_and_Adaptation_2026-10-27.md`: original dataset priorities and local adapter requirements. No datasets were downloaded.
5. `figures/`: exact scientific plots from saved evidence, not generated illustrations.

## Scope of the evidence

Six **untrained, frozen** spatial cores: CNN2D/U-Net × hard LIF/ALIF/continuous recursion. They inherit the user's actual backbone and Cell implementations. The old one-update vector checkpoint is not used as an image reconstruction model. Original random native heads remain unchanged; fitted readouts are explicit separate adapters.

Main: 36 selected image decoders, 36 independent natural-state probes, 12 auxiliary factor-sign probes; 228 ridge candidate solves. Development256 and confirmation512 independent scene pairs, each 144 state four-group cells and 288 last/mean view units. All main choices frozen before generating both cohorts. No SNN optimizer updates, no author pretrained models, no A40, no audit_test; formal benchmark remains 0/435.

Supplementary analyses use the first128 confirmation scenes, except256 new changed-texture OOD scenes. After main results, a separately labeled exploratory summary comparison fits 36 decoders /108 candidate solves and evaluates512 new scenes; equal decoder coefficients do not equal total cost. All raw unfavorable results remain.

Verification: 49 tests, 20 known-answer control families; 9216 main scene-loss arrays and3456 bounded intervals recomputed; all576 saved main view-prefix units replayed (eight scenes in original batch64), payload/state/prediction gaps0. Also480 supplementary loss arrays,38 SSIM arrays and36 matched-summary arrays checked. These are not independent trained-model repetitions.

## Runtime

Execution used Python3.12, torch2.14.1+cpu, NumPy2.3.5, SciPy1.17.0, Matplotlib3.10.8, scikit-image0.26.0. Install `requirements-runtime.txt` in your own environment. Exact bitwise replay can depend on numerical libraries/platform. Included random initialization snapshots match the original frozen parameter hashes and improve portability; they contain no newly trained core.

Run from this directory. Inherited paths resolve relative to `bridge45.py`. Generated experiment directories refuse to overwrite existing outputs. All core computations set one CPU thread.

## Audit the bundled evidence

```bash
python -m unittest discover -s tests -p 'test_spatial45.py' -v
python run_spatial_controls45.py --output review_controls.json
python verify_spatial45_artifacts.py --output review_verification.json
```

The verifier checks all254 unchanged parent files, fitted-bank/model hashes, all main saved losses and intervals, all saved main prefixes and selected supplementary contracts. It does not claim to regenerate every original model forward pass or all external author experiments.

## Regenerate using the included frozen bank

```bash
python run_spatial_main45.py --pool development --output review_runs/development
python run_spatial_main45.py --pool confirmation --output review_runs/confirmation
python run_spatial_supplements45.py --output review_runs/supplements
python run_matched_summaries45.py --output review_runs/matched_summaries
python run_spatial_quality45.py --output review_runs/quality
python resource_accounting45.py --output review_resources.json
```

The supplements/quality scripts intentionally use the included frozen bank. The matched-summary script fits its own separate exploratory adapters using the original independent train/val and new evaluation scenes. `plot_spatial45.py` reads bundled evidence and writes figures; run it in a disposable copy if preserving archive bytes.

## Refit offline readers

```bash
python fit_spatial_bank45.py --output review_runs/refitted_bank
python run_spatial_main45.py --pool development --fit-directory review_runs/refitted_bank --output review_runs/refitted_development
```

Core parameters remain fixed. Do not mix refitted banks with bundled confirmation or supplementary reports. Exact IDs, masks, patch methods and budgets are in `evidence/fitted/freeze_contract.json`.

## Inspect selected author code

```bash
python fetch_pinned_sources45.py --directory review_author_sources
python audit_spatial_authors45.py --source-root review_author_sources --output review_author_audit.json
```

Public GitHub access is needed only for exact text files. Both Git blob and SHA256 are checked. Original selected functions/methods are executed on artificial inputs; EVSNN uses a declared state fixture rather than its full SpikingJelly class. No external datasets or weights are fetched by these scripts. Official DNS/SSMT code links not retrieved remain recorded gaps.

## Local real-data boundary

`real_reconstruction_contract45.py` adapts an already converted U-TILISE-style sample; it is not a complete raw HDF5/GeoTIFF/ROS/event-h5 loader. It requires explicit verified clean targets, source identity, bands, binary mask semantics and observation availability. `mode='causal'` and `mode='offline'` are separate contracts; missing semantic compatibility does not become an H mask automatically.

```bash
python real_reconstruction_contract45.py --sample YOUR_CONVERTED_SAMPLE.npz --target-index 5 --source-id ORIGINAL_ROI_ID --verified-clean-target
```

The command assumes a genuinely verified clean target and `availability_days` in the local NPZ. The download/adapter report explains what must be checked once actual data is available.

## Files and attribution

`references/project_provenance.json`: unchanged parent and source archive identity. `references/original_plans/`: earlier user plans, preserved. `references/selected_author_sources.json`:19 fixed author texts from9 repositories; full author source copies are not redistributed. `references/l45_plan_coverage.json`: original plan/reference coverage. `module_status.json`: exact counts and scope. `MANIFEST_SHA256.json`: all package member hashes except the manifest itself.

Historical parent results are not new L4.5 evidence. Development interface completeness, scientific formal confirmation and publication novelty are distinct. Real-data auditing, end-to-end sufficiently trained cores, nonlinear reader families, training seeds, natural motion/change, deployment and same-total-budget architecture gains remain research to execute.
