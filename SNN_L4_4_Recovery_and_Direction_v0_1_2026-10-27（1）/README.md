# SNN L4.4 — Recovery, patching and direction verification

2026-10-27, v0.1. Chinese research reports and executable development evidence. The original six research layers and L4.1–L4.5 are preserved.

## Start here

1. `SNN_L4_4_Deep_Review_and_Execution_2026-10-27.md`: purpose, eight subsections, actual findings, literature adaptations and paper organization.
2. `SNN_L4_4_Mathematical_Specification_2026-10-27.md`: exact identities, conditional proofs, counterexamples and code correspondence.
3. `SNN_L4_4_Literature_and_Code_Map_2026-10-27.md`: all three original planned references plus four direct neighbors, fixed author sources and unresolved code gaps.
4. Updated Master, whole-L4 research framework and mathematical framework at this directory's root.
5. `figures/L4_4_recovery_evidence.png`: four-panel development summary.

## What was actually executed

The inherited checkpoint has **one original optimizer update**. No SNN weights were trained this round; formal benchmark completion remains **0/435**. CPU only, no A40, no audit_test, no external datasets or pretrained author models. Offline diagnostic probes were fitted separately and are not SNN training.

Main evaluation: 38 recovery contexts × 10 patch methods × two disjoint scene cohorts (1024 development and 2048 confirmation pairs), giving 760 four-group cells. Supplementary experiments reuse the first 256 confirmation pairs. They are not additional independent cohorts.

Additional evidence: 72 event-policy four-group cells; 30 joint-owner contexts / 120 subset conditions; 63 binary-edge four-group cells; 12 actual path-reblocking cases; 72 event-geometry dose conditions; 240 untrained snapshot interfaces / 1920 scope conditions; 38 unit tests; 14 known-answer control families. Untrained classification/reconstruction interfaces do not establish trained task performance.

Semantic probe readability can return close to its clean level while native task benefit remains unqualified. No original MSE recovery ratio passed the prespecified damage floor. Positive risk interaction alone is not recovery success. All four risks and prediction interaction are retained.

## Environment

Python 3.11+ with `requirements-runtime.txt`. Execution used torch 2.14.1+cpu, NumPy 2.3.5, SciPy 1.17.0 and Matplotlib 3.10.8; these observed versions are recorded, not a requirement to obtain that exact build. Floating-point replay tolerances can depend on platform. Single CPU thread is set inside execution scripts.

Run from this directory after extracting the archive. All inherited imports are relative to `bridge44.py`; the unchanged `parent_l43/` dependency tree and checkpoint are included. No absolute path in the current workstation is needed. Generated evidence directories refuse to overwrite existing results.

## Verify without refitting

```bash
python -m unittest discover -s tests -p 'test_recovery44.py' -v
python run_recovery_controls.py --output review_controls.json
python verify_l44_artifacts.py --output review_verification.json
```

The full verifier checks 171 unchanged inherited files, checkpoint/fitted-bank hashes, all 9300 main scene-loss arrays, 378 binary repair-loss arrays, 9232 bounded intervals, and replays the first 16 scenes of **all 760 main cells** in the original batch size 128. Saved payloads, predictions and repaired states reproduced with maximum absolute difference 0 on the execution platform. This is a limited-prefix execution audit plus a full saved-array/statistic audit, not regeneration of every scene's model forward pass.

## Regenerate main and supplementary analyses

The commands below use the included frozen fit bank and generate fresh output directories. They do not train the SNN or modify the archived evidence.

```bash
python run_recovery_main.py --pool development --output review_runs/development
python run_recovery_main.py --pool confirmation --output review_runs/confirmation
python run_recovery_supplements.py --output review_runs/supplements
python run_scoped_snapshot_grid.py --output review_runs/scoped_snapshot_grid.json
python run_binary_recovery.py --output review_runs/binary_recovery
python run_path_reblocking.py --output review_runs/path_reblocking
python run_event_geometry.py --output review_runs/event_geometry
```

`run_event_geometry.py` was added after the main experiments as a diagnostic. No patch or alpha was selected from its outcomes. `plot_l44.py` reads the bundled `evidence/` and writes the bundled figure; to preserve the evidence package, run it in a disposable extracted copy.

## Refit diagnostic bank, if desired

```bash
python fit_recovery_bank.py --output review_runs/refitted_bank --steps 400
python run_recovery_main.py --pool development --fit-directory review_runs/refitted_bank --output review_runs/refitted_development
```

This fits offline probes and new patch-time erasers, not SNN weights. The main program accepts `--fit-directory`; supplementary scripts intentionally read the bundled frozen bank. A refitted bank should not be mixed with bundled confirmation/supplement evidence. The exact archived fit was frozen before generating both main pools.

Fit allocation: geometry_fit 2048, probe_train 1024, probe_val 256; 14 inherited stable damage erasers plus 14 new patch-time erasers, 24 selected independent validation probes, 36 ridge candidate solves and 24 MLP fits / 9600 offline updates. Exact disjoint IDs and choices are in `evidence/fitted/freeze_contract.json`.

## Recheck selected author code

```bash
python fetch_pinned_sources_l44.py --directory review_author_sources
python audit_recovery_authors.py --source-root review_author_sources --output review_author_audit.json
```

The fetch needs public GitHub access and verifies both Git blob and SHA256. It retrieves only six fixed text files, not models/datasets. The audit executes canonical DecodeShare definitions and selected ROME causal-tracing definitions on artificial Torch networks. It does not reproduce their complete paper results. Exact unavailable repositories remain documented gaps.

## Files and integrity

`references/project_provenance.json`: inherited archive identity and every unchanged parent file. `references/selected_author_sources.json`: exact author source versions. `references/l44_plan_coverage.json`: original plan coverage and new adaptations. `module_status.json`: counts and execution limits. `MANIFEST_SHA256.json`: archive member integrity (all members except the manifest itself). The ZIP is CRC-checked and every listed member is hashed at packaging.

Historical evidence inside `parent_l43/` remains labeled as prior work. It is never added to the new L4.4 experiment counts. Review corrections (delayed damaged-state validation and MSE-only ratio metadata) are separately recorded; raw outcomes and fixed patch choices were retained.

L4.4 development analysis and reference implementation are delivered. Formal sufficiently trained multi-model, multi-seed, native classification/reconstruction and architecture-improvement evidence remains future research; L4.5 has not been executed by this delivery.
