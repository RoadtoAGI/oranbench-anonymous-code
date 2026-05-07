# Weibo Dryad Cascade Mapping

## Source Fields

The mapping targets the Dryad Weibo cascade dataset identified by `dryadcascade` in `references.bib`. The exact prepared schema is finalized after timestamp inspection.

| Source concept | Target prepared field |
|---|---|
| Original post identifier | `post_id` |
| Root post timestamp | `root_timestamp` |
| Repost / cascade event timestamp | `event_timestamp` |
| Event user or node id | `event_user_id` when available |
| Early observation window | `prefix_window` |
| Held-out prediction window | `tail_window` |
| Static post features | `post_features` when available |
| Final cascade size | `cascade_size` |

## Exposure-Unit Schema

| Field | Type | Description |
|---|---|---|
| `unit_id` | string | Cascade event id |
| `creative_id` | string | Original post id |
| `event_timestamp` | timestamp | Repost or cascade event time |
| `event_user_id` | string | Event node or user id when available |
| `time_since_root` | float | Elapsed time from root post |
| `prefix_features` | object | Events observed before the prediction cutoff |
| `exposure_regime` | string | `organic_cascade` |

This layer supports point-process and cascade calibration baselines when event timestamps are present.

## Creative-Aggregate Schema

| Field | Type | Description |
|---|---|---|
| `creative_id` | string | Original post id |
| `root_timestamp` | timestamp | Root post time |
| `post_features` | object | Available static post-level features |
| `prefix_size` | integer | Number of events in the early window |
| `prefix_rate` | float | Early event rate |
| `cascade_size` | integer | Final or horizon-limited cascade size |
| `tail_size` | integer | Events after the prefix cutoff |
| `censoring_window` | object | Observation and forecast windows |

## Prediction Targets

The primary target is cascade size or tail size at a fixed horizon. When event timestamps support point-process estimation, secondary metrics include held-out log-likelihood, tail MAPE or sMAPE, and branching-ratio diagnostics.

## Applicable Baselines

Simple baselines: constant-rate Poisson, prefix-size heuristic, and linear regression on post/prefix features.

ML baselines: LightGBM or MLP for cascade-size prediction.

Diffusion baselines: Hawkes MLE and SEISMIC-style predictors; DeepCas-style neural baselines are included when preprocessing time permits.

## OranSim Reference-Label Role

Weibo Dryad ships its mapping document and schema definitions in v0.1 as a stretch entry. The dataset is **not** used as a calibration anchor or validation anchor for the OranSim reference-label layer. Cascade-size baseline rows are completed before camera-ready, contingent on Dryad timestamp-schema inspection.

## Limitation

Weibo Dryad is an organic cascade dataset. It contributes diffusion outcome prediction and temporal calibration only when v0.1 schema inspection is complete; controlled exposure comparisons are supplied by the other datasets.
