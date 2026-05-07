# KuaiRand-Pure Mapping

## Source Fields

The mapping uses KuaiRand-Pure random-exposure interaction files prepared by `backend/scripts/prepare_kuairand.py`.

| Source concept | Prepared field |
|---|---|
| User identifier | `user_id` |
| Video / creative identifier | `item_id` |
| Action identifier | `action_id` (= `item_id` in v0.1) |
| Timestamp | `timestamp` |
| Logging propensity | `propensity` |
| Click feedback | `reward_click` |
| Like feedback | `reward_like` |
| Long-view feedback | `reward_long_view` |
| Watch-time feedback | `reward_watch_time` |

The random-exposure slice uses known uniform propensity \(p_i=1/7583\), corresponding to the KuaiRand-Pure random candidate pool.

## Exposure-Unit Schema

| Field | Type | Description |
|---|---|---|
| `unit_id` | string | Stable row identifier derived from source row order |
| `creative_id` | string | Video id, copied from `item_id` |
| `audience_id` | string | User id, copied from `user_id` |
| `action_id` | string | Exposed video id |
| `timestamp` | source type | Exposure time |
| `propensity` | float | Known logging probability |
| `reward_click` | float | Click indicator |
| `reward_like` | float | Like indicator |
| `reward_long_view` | float | Long-view indicator |
| `reward_watch_time` | float | Watch-time outcome |
| `exposure_regime` | string | `platform_randomized` |

This layer supports IPS, SNIPS, DR, Switch-DR, and sequential or tabular outcome models.

## Creative-Aggregate Schema

| Field | Type | Description |
|---|---|---|
| `creative_id` | string | Video id |
| `n_exposures` | integer | Number of random exposures |
| `n_users` | integer | Number of distinct exposed users |
| `first_timestamp` | source type | Earliest observed random exposure |
| `last_timestamp` | source type | Latest observed random exposure |
| `audience_feature_summary` | object | Available aggregate user statistics; empty when user side features are absent from the prepared slice |
| `click_rate` | float | Mean `reward_click` |
| `like_rate` | float | Mean `reward_like` |
| `long_view_rate` | float | Mean `reward_long_view` |
| `mean_watch_time` | float | Mean `reward_watch_time` |

## Prediction Targets

Primary targets are `long_view_rate` and `reward_long_view`. Secondary targets are click, like, and watch time. Exposure-unit results use policy-value estimators; aggregate results use outcome prediction over creative-level rates.

## Applicable Baselines

Simple baselines: logged mean, creative popularity, linear regression on aggregate features.

ML baselines: LightGBM, MLP over user/video features, SASRec on exposure sequences, and the post-abstract action-conditioned Transformer outcome model if training metrics are available.

OPE baselines: IPS, SNIPS, DR, Switch-DR, and DR-OS-style shrinkage where supported by the implementation.

## OranSim Reference-Label Role

KuaiRand-Pure serves as the **public exposure-regime validation anchor** for the OranSim reference-label layer (proposal §3.5). The known uniform propensity \(p_i = 1/7583\) and the rows / users / items totals from the prepared manifest support the simulator's assumption that exposure-regime fields are well-defined under a randomized policy. The same slice instantiates the C1 off-policy baselines, so the simulator's input-side exposure assumption is grounded in a public benchmark that is also evaluated under the public protocol layer.

The dataset is **not** used as an outcome-scale calibration source for the simulator; the simulator's outcome scale is calibrated separately under stated assumptions (proposal §3.5, Appendix B).

## Limitation

KuaiRand-Pure evaluates platform-randomized video exposure on a Chinese short-video platform. Its propensity structure is useful for OPE and outcome-regression diagnostics, while advertiser budget allocation and campaign-level incrementality require a different exposure design.
