# Source and dependency attribution

Project dependencies under `parent_l42/` are carried unchanged from the user's previous L4.2/L4.1 and selected L3.7 artifacts. Their provenance is in `references/project_provenance.json`; their old evidence is historical, not newly produced L4.3 evidence.

The new mathematical reference implementation adapts published LEACE/mean-projection and established linear algebra, risk and probability methods. It does not claim those formulas as newly invented. The fixed covariance metric, finite SNN checks, experiment contract and interpretation are described with assumptions in the accompanying reports.

Full external author source files are not redistributed in this package. `references/selected_author_sources.json` gives repository, commit, Git blob, SHA256 and source URLs for the exact files inspected. `fetch_pinned_sources_l43.py` retrieves only those public files for the local source audit. Original LEACE classes and selected AST definitions were executed without editing their definitions; these limited checks do not reproduce author training or paper datasets. Consult each original repository's license before redistributing its source.

Core references and original repository URLs are enumerated in the literature map and BibTeX file. The six original Master reference identities are preserved.
