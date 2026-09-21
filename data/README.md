# Data

Maintained code should not rely on undocumented mutable remote data.

Historical datasets from the original notebook are stored under `data/legacy/` and tracked with Git LFS where applicable. They are retained for provenance rather than treated as canonical current inputs.

New datasets should include, at minimum:

- source and retrieval date;
- licence or redistribution constraints;
- schema and units;
- transformations applied;
- checksums or version identifiers when practical.

Generated outputs should not be committed here unless they are small, deterministic fixtures required by tests.
