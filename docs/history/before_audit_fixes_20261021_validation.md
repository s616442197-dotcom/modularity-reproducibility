# Current validation — 2026-10-21

Current local working tree passed result-based figure rendering, six reduced training/analysis smoke cases, scalar-weight cost extraction smoke, and published-statistic checks. Fig.5 has 180 runs of width 300, 150 paired observations and pooled r = 0.7835066365; Fig.6b–c has 360 evaluations. Cost extraction covers 194700 scalar weights per model.

Checks ran in a copied repository with a Python audit hook denying the original source and repository paths. This is not an OS sandbox or a fresh dependency installation. The existing modularity_env was used. Full fresh training of all 923 runs was not repeated. TeX was compiled separately, including an unpacked submission ZIP, rather than within the hook-isolated test.

See [current_validation.json](current_validation.json) and [current_source_manifest.json](current_source_manifest.json). Remote GitHub state has not been updated or certified by these checks. Earlier checks are retained in [historical validation](history/validation_before_20261021.md).
