"""
MSE, 線形相関，Spearman相関，Kendall_tauを計算する関数群
"""

import numpy as np
from scipy import stats


def _validate_inputs(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    入力配列からnan/infを含む要素を除外する
    """
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        raise ValueError("All values are nan or inf")
    return y_true[mask], y_pred[mask]


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    平均二乗誤差 (Mean Squared Error) を計算する関数
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    ピアソン相関係数を計算する関数
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    corr, _ = stats.pearsonr(y_true, y_pred)
    return float(corr)


def spearman_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    スピアマンの順位相関係数を計算する関数
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    corr, _ = stats.spearmanr(y_true, y_pred)
    return float(corr)


def kendall_tau(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    ケンドールの順位相関係数を計算する関数
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    corr, _ = stats.kendalltau(y_true, y_pred)
    return float(corr)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    分類精度を計算する関数

    Args:
        y_true: 正解ラベル（整数のインデックス）
        y_pred: 予測ラベル（整数のインデックス）

    Returns:
        分類精度（0.0 ~ 1.0）
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")

    if len(y_true) == 0:
        return float("nan")

    correct = np.sum(y_true == y_pred)
    return float(correct / len(y_true))
