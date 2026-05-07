# Open Bandit Dataset Bundled Mapping

## Source Fields

The v0.1 artifact uses the OBP bundled Open Bandit Dataset sample loaded through `obp.dataset.OpenBanditDataset`.

| Source concept | OBP field |
|---|---|
| User / context features | `context` |
| Recommended item / action | `action` |
| Position | `position` |
| Click reward | `reward` |
| Logging propensity | `pscore` |
| Behavior policy | `behavior_policy` selected at load time (`random` or `bts`) |
| Action context | `action_context` when available |

## Exposure-Unit Schema

| Field | Type | Description |
|---|---|---|
| `unit_id` | string | Stable row identifier within behavior-policy sample |
| `creative_id` | string | Recommended item id, copied from `action` |
| `audience_features` | vector | OBP context vector |
| `action_id` | integer | Recommended item id |
| `position` | integer | Recommendation slot |
| `propensity` | float | Logged propensity `pscore` |
| `reward_click` | integer | Click reward |
| `logging_policy` | string | `random` or `bts` |
| `exposure_regime` | string | `logged_production_policy` |

This layer supports OPE estimators and outcome-regression baselines under logged-bandit evaluation.

## Creative-Aggregate Schema

| Field | Type | Description |
|---|---|---|
| `creative_id` | string | Recommended item id |
| `logging_policy` | string | Behavior-policy slice |
| `n_exposures` | integer | Number of logged recommendations |
| `audience_feature_summary` | object | Summary statistics over OBP context vectors |
| `position_summary` | object | Position distribution |
| `click_rate` | float | Mean click reward |
| `mean_propensity` | float | Mean logged propensity |

## Prediction Targets

The exposure-unit target is click reward under logged recommendation. OPE results evaluate synthetic target policies with reported support diagnostics. The aggregate target is item-level click rate within a behavior-policy slice.

## Applicable Baselines

Simple baselines: logged mean, item popularity, position-adjusted mean, linear regression on context and action features.

ML baselines: LightGBM, MLP over context/action features, SASRec-style sequential model when ordered exposure sequences are available.

OPE baselines: IPS, SNIPS, DR, Switch-DR, and DR-OS through OBP.

## OranSim Reference-Label Role

OBD is **not** used to calibrate the OranSim reference-label layer. It serves as a **public logged-bandit-protocol sanity check** that exercises the two-layer interface and the OBP estimator family under multiple behavior policies (random and BTS). The per-(logging policy, target policy) ESS and clipping diagnostics reported in §5 demonstrate the protocol's behavior under a logging policy that differs from the target, complementary to the C1 randomized-exposure anchor used by the simulator.

## Limitation

The v0.1 artifact uses the OBP bundled sample rather than the full OBD release. Results are suitable for interface and estimator checks; full-scale OBD reporting requires replacing the bundled sample with the hosted production release.
