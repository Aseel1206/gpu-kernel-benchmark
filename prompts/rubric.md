# Prompt-quality scoring rubric

Used to fill `prompt_quality_score` in each kernel's `*.meta.json`. Score 1-5 per axis, record the
total (max 20) plus the per-axis breakdown in `notes`.

| Axis | 1 (poor) | 3 (ok) | 5 (excellent) |
|---|---|---|---|
| Specificity | Vague op description, no shapes/dtypes | Shapes/dtypes given, some ambiguity remains | Exact shapes, dtype, memory layout, and edge cases (e.g. non-divisible block sizes) specified |
| Constraints | No performance/correctness constraints stated | Mentions "should be fast" | States target tolerance, forbids calling the torch equivalent, specifies expected complexity class |
| Context | No hardware/framework context | Names CUDA/PyTorch | Names CUDA/compute-capability expectations, target GPU family, and any launch-config tuning expectation |
| Output format | Unspecified | Asks for code | Pins exact function name/signature and "code only, no prose" |

This rubric is applied to the *same* base prompt every time by default (full-matrix runs use prompt
quality held constant so it isn't a confound). If you deliberately vary prompt quality as a side
experiment, save each variant as `prompts/<kernel>_prompt_v2.md`, `v3.md`, etc., and record which
`prompt_id` was used in every `*.meta.json`.
