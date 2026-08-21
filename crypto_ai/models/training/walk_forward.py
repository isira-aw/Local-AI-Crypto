"""
Walk-forward split generator (Section 13 / Phase 9).

What is it?
    Produces an EXPANDING series of (train, embargo, validation) index
    ranges over a chronologically-ordered dataset, plus one final,
    completely held-out test range that no fold ever touches.

Why is it needed?
    A single train/test split can get lucky (or unlucky) on one
    period. Walk-forward validation checks whether a strategy holds up
    across several different time periods — and the embargo gap
    between train and validation prevents a label computed from
    "future" data from leaking across the split boundary.

    FOLD 1: TRAIN -> EMBARGO GAP -> VALIDATE
    FOLD 2: TRAIN (extended) -> EMBARGO GAP -> VALIDATE
    ...
    FINAL: TRAIN -> TEST (fully held out) -> PAPER TRADING

How does it work?
    Given `n_rows` chronologically-ordered samples:
      1. Reserve the last `final_test_bars` rows as the final test set
         — never used by any fold below.
      2. Fold i's training window is [0, train_end_i) where
         train_end_i grows by `validation_bars` each fold (expanding
         window, not sliding — more data is strictly better for a
         model as long as it's still in-sample).
      3. Skip `embargo_bars` rows after the training window ends.
      4. The next `validation_bars` rows are fold i's validation set.
      5. Stop once there isn't enough remaining data (before the final
         test set) for another full validation window.

    `embargo_bars` MUST be >= the label horizon (settings.yaml:
    labeling.horizon_bars) — otherwise a label in the validation set
    could have been computed using price data that overlaps the
    training window, which is exactly the leakage this is meant to
    prevent.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WalkForwardFold:
    fold_index: int
    train_start: int
    train_end: int  # exclusive
    val_start: int
    val_end: int  # exclusive

    @property
    def train_slice(self) -> slice:
        return slice(self.train_start, self.train_end)

    @property
    def val_slice(self) -> slice:
        return slice(self.val_start, self.val_end)


@dataclass
class WalkForwardPlan:
    folds: list[WalkForwardFold]
    final_test_start: int
    final_test_end: int

    @property
    def final_test_slice(self) -> slice:
        return slice(self.final_test_start, self.final_test_end)


def generate_walk_forward_folds(
    n_rows: int,
    n_folds: int,
    min_train_bars: int,
    validation_bars: int,
    embargo_bars: int,
    final_test_bars: int,
) -> WalkForwardPlan:
    if embargo_bars < 0:
        raise ValueError("embargo_bars must be >= 0")

    final_test_start = max(0, n_rows - final_test_bars)
    usable = final_test_start  # folds may only use rows before the final test set

    folds: list[WalkForwardFold] = []
    train_end = min_train_bars
    for i in range(n_folds):
        val_start = train_end + embargo_bars
        val_end = val_start + validation_bars
        if val_end > usable or train_end > usable:
            break
        folds.append(
            WalkForwardFold(fold_index=i, train_start=0, train_end=train_end, val_start=val_start, val_end=val_end)
        )
        # Expanding window: the next fold's training set grows to include
        # this fold's validation data too (it is now "past" data).
        train_end = val_end

    return WalkForwardPlan(folds=folds, final_test_start=final_test_start, final_test_end=n_rows)


def assert_no_leakage(plan: WalkForwardPlan, embargo_bars: int, label_horizon_bars: int) -> None:
    """
    Sanity-checks a plan: every fold's validation window must start at
    least `label_horizon_bars` rows after its training window ends,
    and no fold may reach into the final test region.
    """
    if embargo_bars < label_horizon_bars:
        raise ValueError(
            f"embargo_bars ({embargo_bars}) must be >= label_horizon_bars "
            f"({label_horizon_bars}) or validation labels can leak training data."
        )
    for fold in plan.folds:
        gap = fold.val_start - fold.train_end
        if gap < label_horizon_bars:
            raise ValueError(f"Fold {fold.fold_index}: gap {gap} < label horizon {label_horizon_bars}")
        if fold.val_end > plan.final_test_start:
            raise ValueError(f"Fold {fold.fold_index} overlaps the final held-out test set")
