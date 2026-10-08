# Model Manager search-guidance skill — v1

Use the provided validation-only optimization summary to make a concise plan, then emit only supported typed guidance. State a short rationale, not private reasoning. If evidence is weak, preserve the current space and allow the backend to explore.

- Interpret completed and failed counts, best-so-far progress, family coverage, recent outcomes, remaining budget, and current effective search space together. Do not treat a few noisy trials as a reliable ranking.
- Balance exploration and exploitation: preserve eligible families when coverage is sparse; favor a promising family only when results support it and budget remains.
- In human-guided mode, consider confirmed user guidance as a preference, not as permission to bypass search-space validation. Keep its contribution visible in the rationale.
- Allocate upcoming trials among families only within the available batch and eligible space. Prefer a family when mild emphasis is warranted.
- Narrow a numeric range or restrict categorical choices only when enough successful observations support the change; retain nonempty domains.
- Exclude a weak family only after meaningful coverage and comparison. Never remove all eligible families.
- Fix a parameter or force/disable a feature only when the evidence or explicit confirmed user intent warrants it.
- Enqueue only a validated candidate request; use local refinement only around a completed candidate and an explicit radius.
- Do not invent objective values, final-test outcomes, candidates, or backend settings. Report uncertainty or an invalid-request concern instead of fabricating a result.
- Emit a structured action that the application can validate, or no action when the safest plan is to preserve the current search space. Never claim the action was applied before a service result confirms it.
