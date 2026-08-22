# Model Training

## What is it?

The automatic pipeline that turns features + labels into a validated,
versioned ML model — `crypto_ai/models/training/pipeline.py`.

## Why is it needed?

So you never have to manually choose hyperparameters, manually check for
overfitting, or manually decide "is this model good enough." The pipeline
enforces a leakage-safe validation procedure and a promotion checklist
before anything becomes a candidate for production use.

## What goes in / comes out?

**In:** a feature DataFrame (`compute_features`) and a label DataFrame
(`compute_labels`), a symbol/timeframe, and a list of candidate algorithms.
**Out:** one or more `ModelVersion` rows in the database (status
`CANDIDATE` or `REJECTED`), each with a saved model file
(`data_store/models/<name>/<version>/model.joblib`) and full walk-forward
results attached — and, if a candidate beats the current champion, a
promotion.

**Start it:** `python run.py train`

## How does it work?

### Candidate models (Section 8)

Three simple, well-understood algorithms, in order of increasing
complexity:

- Logistic Regression
- Random Forest
- Gradient Boosting (histogram-based — see "Training time" below)

Neural networks are deliberately **not** the default. Section 8 of the
design document is explicit: a complex model is only worth using if it
actually beats a simple one on this data — and simple models are much
easier to reason about, retrain quickly, and run on a CPU-only machine.
If you want to add XGBoost/LightGBM, see `crypto_ai/models/training/models.py`
— the pipeline doesn't care which algorithm produces a scikit-learn-style
`Pipeline` with `predict_proba`.

### Walk-forward validation, not a single train/test split

```
FOLD 1: TRAIN            -> EMBARGO GAP -> VALIDATE
FOLD 2: TRAIN (extended) -> EMBARGO GAP -> VALIDATE
...
FINAL:  TRAIN            -> TEST (never touched by any fold) -> paper trading
```

Each fold's training window **expands** to include the previous fold's
validation data (more history is better, as long as it's still in the
past relative to what's being validated). The **embargo gap** between
training and validation is at least as long as the label horizon — this is
what stops a validation-set label (which is computed from a *future* price)
from ever having been influenced by information the model already saw
during training. This is checked in code
(`models/training/walk_forward.py:assert_no_leakage`) and covered by
dedicated tests.

The **final test set** is never used by any fold, for anything — it's the
one piece of held-out data whose result actually means something once
training is done.

### Promotion criteria (Section 15) — not just accuracy

A model only becomes a `CANDIDATE` if it passes on enough folds (default:
80%, see `models.promotion_criteria.min_folds_passed_pct` in
`settings.yaml`) **and** if there were enough folds to begin with
(`min_folds_required`, default 3).

That second condition matters more than it looks. With limited history the
fold generator can only produce one fold — and "1 of 1 folds passed" is
100%, which would sail past a percentage-based threshold while proving
nothing about robustness across market regimes. A run with too few folds is
therefore rejected outright, with a message telling you to collect more
history (or lower `walk_forward.min_train_bars` / `validation_bars`).

"Passing a fold" requires **all** of:

- Precision above a minimum (directional accuracy on BUY/SELL calls)
- Maximum drawdown within a limit
- A minimum Sharpe ratio (risk-adjusted return, from a realistic backtest of
  that fold's signals — run at the **same position size and stop/target
  levels the risk engine would actually enforce live**, read from
  `risk.yaml`, so promotion is never decided on an exposure the system is
  not allowed to take)
- A minimum number of trades (so the numbers aren't from 2 lucky trades)

Failing any fold criterion counts against the model. A model that clears
accuracy but has huge drawdown, or clears drawdown but barely trades, does
**not** get promoted.

### Model registry (Section 24)

Every attempt — passed or failed — is saved with a version label
(`model_001`, `model_002`, ...) and a status:

```
TRAINING -> TESTING -> CANDIDATE -> CHAMPION
                                  -> REJECTED
CHAMPION -> RETIRED (when a better candidate is promoted)
```

The **champion** is the model currently used for live predictions/paper
trading. A new candidate only replaces it if it beats the champion's
held-out test Sharpe ratio — never automatically on accuracy alone, and
never without a comparison against what's already running
(`maybe_promote_champion()` in `pipeline.py`). Rollback to the previous
champion is possible via `models/registry/registry.py:rollback_to_previous_champion`.

## Training time and the data window

Walk-forward training does 6 fits per algorithm (5 folds + 1 final model),
so per-fit cost matters a lot. Measured on this project's own data shape
(20k rows x 30 features, 4-core CPU):

| Estimator | Per fit | 6 fits |
|---|---|---|
| Logistic Regression | 0.1s | negligible |
| Random Forest (200 trees, depth 6) | 6.3s | ~40s |
| `GradientBoostingClassifier` (exact) | 84.4s | ~8.4 min |
| `HistGradientBoostingClassifier` | 18.4s | ~1.8 min |

`gradient_boosting` therefore uses the **histogram-based** implementation,
which is what sklearn itself recommends above ~10,000 samples. The exact
one is still selectable as `gradient_boosting_exact`.

Two knobs control how long a run takes (Section 53):

- `resource_limits.max_training_bars` (**DATA_WINDOW**, default 60000) —
  caps how much history one run uses, keeping the **most recent** bars.
  The full 2-year 5m history is ~210k bars, which takes hours; 60k bars is
  ~7 months and still supports all 5 walk-forward folds. Set to `null` to
  use everything.
- `resource_limits.max_workers` — parallelism for Random Forest.

Don't run training and the local LLM at full load at the same time on a
modest machine; they compete for the same CPU and RAM.

## Multiple-testing correction

If you train 3 algorithms and promote the best, that winner's Sharpe is
biased upward purely because three things were tried — the maximum of N
noisy estimates exceeds the truth even when every variant is worthless.

This is enforced, not just reported: `models/evaluation/multiple_testing.py`
computes a Deflated Sharpe Ratio (Bailey & Lopez de Prado) and
`maybe_promote_champion()` **refuses to promote** any candidate that cannot
clear `models.promotion_criteria.min_deflated_sharpe_probability`
(default 0.95). A blocked promotion is logged as
`promotion_blocked_by_multiple_testing` with the full numbers.

Two details that matter for correctness:

- **Trial count is cumulative.** It counts the algorithms in this run *plus*
  every variant previously registered for the model. If you have trained
  twenty models over six weeks and are promoting the best, the
  multiple-testing surface is twenty, not three.
- **Units are converted.** The standard error in the DSR formula is defined
  for a per-period Sharpe, while this project reports annualized ones.
  Treating an annualized value as per-period inflates DSR toward 1.0 and
  would silently defeat the check, so the value is de-annualized first.

With fewer than four trials the cross-sectional spread of trial Sharpes is
dominated by the winner itself — circular, since a strong result would
inflate its own bar — so below that count the standard estimation-noise
assumption is used instead.

## Too-good-to-be-true detection (Section 47)

`backtesting/metrics.py:suspicious_result_flags` checks for red flags: too
few trades, an implausibly high Sharpe ratio, a suspiciously high win rate,
an extreme total return, or a big gap between train and test performance.
These flags are attached to every backtest/training result and shown on the
dashboard. **If you see them, don't ignore them** — they usually mean
overfitting or a subtle data leak, not a genuinely great strategy.

## The local LLM's role here

None, directly. The LLM never chooses hyperparameters or decides which
model to promote — see `docs/CRYPTO_BASICS.md` and Section 18 of the design
document. It only explains an already-made decision in plain language
(`llm/analyst.py`), and only if `LLM_ENABLED=true`.

## Troubleshooting

- **"only N walk-forward fold(s) available, need >= 3"**: you don't have
  enough history for a meaningful robustness check. Download more data —
  the defaults need roughly 44k bars (about 5 months of 5m candles) to
  produce the full 5 folds.
- **"No model passed walk-forward validation"**: normal, especially with
  limited history. Try downloading more data, or (for experimentation only)
  loosen `models.promotion_criteria` in `settings.yaml` — but understand
  that loosening criteria doesn't create real edge, it just lowers the bar.
- **Training is slow**: reduce `resource_limits.max_workers`, or train on
  fewer walk-forward folds while iterating, then run the full config before
  trusting a result.
