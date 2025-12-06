"""
MSE, 線形相関，Spearman相関，Kendall_tauを計算する関数群
"""


def mse(y_true, y_pred) -> float:
    """
    平均二乗誤差 (Mean Squared Error) を計算する関数
    """
    ...


def pearson_correlation(y_true, y_pred) -> float:
    """
    ピアソン相関係数を計算する関数
    """
    ...


def spearman_correlation(y_true, y_pred) -> float:
    """
    スピアマンの順位相関係数を計算する関数
    """
    ...


def kendall_tau(y_true, y_pred) -> float:
    """
    ケンドールの順位相関係数を計算する関数
    """
    ...
