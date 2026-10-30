# Method Notes

SLA-Track connects first-frame instance identity to the current target state through a shared visual-language representation and three localization stages.

## Implementation map

| Component | Implementation |
|:--|:--|
| Shared encoder and model integration | `lib/models/sutrack/sutrack.py` |
| ISIA | `InstanceSpecificIdentityAnchor` in `lib/models/sutrack/sla_v3.py` |
| LSC | `LanguageConditionedSpatialCalibration` in the same file |
| SGCR | `SemanticGeometricCandidateResolution` in the same file |
| Module composition | `GroundedSLA` in the same file |
| Training objectives | `lib/train/actors/sutrack.py` |
| Online tracking | `lib/test/tracker/sutrack.py` |

## Tensor flow

| Representation | Shape |
|:--|:--|
| Native search tokens | B × 196 × 512 |
| Static first-frame template tokens | B × 49 × 512 |
| Encoded language / identity vector | B × 512 |
| Identity spatial field | B × 196 |
| Calibrated search tokens | B × 196 × 512 |
| CENTER response / size / offset | B × 1/2/2 × 14 × 14 |
| Candidate entries | B × 10 |
| Final normalized center-format box | B × 4 |

**ISIA** pools the annotated first-frame target and complementary context with fractional token coverage. Language attends to their typed summaries; a parallel target–context contrast branch contributes to the identity correction.

**LSC** computes an identity-conditioned cosine field, applies a temperature of 0.15, scales softmax weights by the search-token count, and clips the gate at 4.0. A projected identity–search interaction supplies the residual feature correction at scale 0.10.

**SGCR** decodes all 196 CENTER locations and pools identity evidence from the original search tokens. It concatenates native Top-5 and semantic Top-5 indices, retaining repeated entries. Candidate scoring uses native and semantic confidence, distractor-relative margins, and box geometry. Parallel bounded geometry residuals refine each candidate before the highest-scoring entry is selected.

The static first-frame annotation supplies ISIA's geometric reference. Calibrated search features feed CENTER, while original search tokens supply SGCR's semantic pooling. The tracker applies the Hann window before final candidate resolution and maps the selected box back to image coordinates.
