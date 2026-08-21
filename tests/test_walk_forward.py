import pytest

from crypto_ai.models.training.walk_forward import (
    assert_no_leakage,
    generate_walk_forward_folds,
)


def test_folds_are_expanding_and_non_overlapping_with_embargo():
    plan = generate_walk_forward_folds(
        n_rows=10000, n_folds=5, min_train_bars=2000, validation_bars=1000, embargo_bars=12, final_test_bars=1000,
    )
    assert len(plan.folds) >= 2
    for i, fold in enumerate(plan.folds):
        assert fold.train_start == 0
        gap = fold.val_start - fold.train_end
        assert gap == 12
        assert fold.val_end <= plan.final_test_start
        if i > 0:
            # Expanding window: each fold's training set is >= previous fold's.
            assert fold.train_end >= plan.folds[i - 1].train_end
            assert fold.train_end == plan.folds[i - 1].val_end


def test_final_test_set_never_touched_by_any_fold():
    plan = generate_walk_forward_folds(
        n_rows=5000, n_folds=10, min_train_bars=1000, validation_bars=500, embargo_bars=12, final_test_bars=500,
    )
    assert plan.final_test_start == 4500
    for fold in plan.folds:
        assert fold.val_end <= 4500
        assert fold.train_end <= 4500


def test_stops_when_not_enough_data_for_requested_folds():
    plan = generate_walk_forward_folds(
        n_rows=3000, n_folds=100, min_train_bars=1000, validation_bars=500, embargo_bars=12, final_test_bars=500,
    )
    assert len(plan.folds) < 100
    assert len(plan.folds) >= 1


def test_assert_no_leakage_passes_for_valid_plan():
    plan = generate_walk_forward_folds(
        n_rows=10000, n_folds=5, min_train_bars=2000, validation_bars=1000, embargo_bars=12, final_test_bars=1000,
    )
    assert_no_leakage(plan, embargo_bars=12, label_horizon_bars=12)


def test_assert_no_leakage_rejects_embargo_smaller_than_horizon():
    plan = generate_walk_forward_folds(
        n_rows=10000, n_folds=5, min_train_bars=2000, validation_bars=1000, embargo_bars=5, final_test_bars=1000,
    )
    with pytest.raises(ValueError):
        assert_no_leakage(plan, embargo_bars=5, label_horizon_bars=12)
