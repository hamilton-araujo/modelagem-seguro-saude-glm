"""
Lift Curve + Gini Coefficient — poder de discriminação atuarial.

Por que existe:
    Um GLM com R² = 0.31 ainda pode ser excelente — desde que ranqueie
    corretamente os segurados de alto vs baixo risco. A métrica que
    importa em tarifação é **discriminação ordinal**, não erro absoluto.

    **Lift por decil:** divide a base em 10 grupos por prêmio previsto;
    o decil mais caro deve ter ~3-5× o custo médio do mais barato.

    **Gini coefficient:** integra a curva de Lorenz dos custos vs prêmios.
    Gini = 0 → modelo aleatório. Gini > 0.30 → bom. Gini > 0.50 → excelente.
"""

import numpy as np
import pandas as pd


def lift_decis(y_real: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    """Devolve tabela de Lift por decil (1=mais barato, 10=mais caro)."""
    df = pd.DataFrame({"real": y_real.values, "pred": y_pred.values})
    df["decil"] = pd.qcut(df["pred"], q=10, labels=False, duplicates="drop") + 1

    grouped = df.groupby("decil").agg(
        n=("real", "count"),
        custo_real_medio=("real", "mean"),
        premio_pred_medio=("pred", "mean"),
        custo_total=("real", "sum"),
    )

    custo_global = df["real"].mean()
    grouped["lift"] = grouped["custo_real_medio"] / custo_global
    grouped["loss_ratio_relativo"] = (
        grouped["custo_real_medio"] / grouped["premio_pred_medio"]
    )

    return grouped.reset_index()


def gini_coefficient(y_real: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Gini sobre a curva de Lorenz dos custos ranqueados pelo prêmio previsto.

    Retorna [0, 1]. > 0.30 = bom modelo atuarial.
    """
    df = pd.DataFrame({"real": y_real, "pred": y_pred}).sort_values("pred")
    cum_real = df["real"].cumsum()
    cum_real = cum_real / cum_real.iloc[-1]
    n = len(df)
    cum_pop = np.arange(1, n + 1) / n

    # Área entre Lorenz e diagonal
    area_below_lorenz = np.trapezoid(cum_real.values, cum_pop)
    return float(2 * (0.5 - area_below_lorenz))


def double_lift(y_real: pd.Series, y_pred_a: pd.Series, y_pred_b: pd.Series,
                n_bins: int = 10) -> pd.DataFrame:
    """
    Double-lift: compara dois modelos. Mostra qual modelo melhor explica
    o sinal residual quando os dois discordam mais (decis extremos do ratio).
    """
    df = pd.DataFrame({"real": y_real.values, "pa": y_pred_a.values, "pb": y_pred_b.values})
    df["razao"] = df["pa"] / df["pb"]
    df["bin"] = pd.qcut(df["razao"], q=n_bins, labels=False, duplicates="drop")

    return df.groupby("bin").agg(
        n=("real", "count"),
        real=("real", "mean"),
        pred_a=("pa", "mean"),
        pred_b=("pb", "mean"),
    ).reset_index()
