"""
MSE, 線形相関，Spearman相関，Kendall_tauを計算する関数群
"""

import numpy as np
from scipy import stats


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    平均二乗誤差 (Mean Squared Error) を計算する関数
    """
    return float(np.mean((y_true - y_pred) ** 2))


def pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    ピアソン相関係数を計算する関数
    """
    corr, _ = stats.pearsonr(y_true, y_pred)
    return float(corr)


def spearman_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    スピアマンの順位相関係数を計算する関数
    """
    corr, _ = stats.spearmanr(y_true, y_pred)
    return float(corr)


def kendall_tau(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    ケンドールの順位相関係数を計算する関数
    """
    corr, _ = stats.kendalltau(y_true, y_pred)
    return float(corr)
