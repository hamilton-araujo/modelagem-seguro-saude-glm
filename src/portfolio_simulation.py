"""
Simulação de portfólio + projeção de Loss Ratio.

Por que existe:
    Treinar GLM em 1.338 vidas é exercício acadêmico. O CFO quer saber:
    aplicado a um portfólio real de 10.000 vidas com perfil demográfico
    realista, qual o Loss Ratio esperado? Atinge target 70%?

    Loss Ratio = Σ Custos / Σ Prêmios

    Premium-to-Risk ratio: quanto da volatilidade vem do prêmio (premium-driven)
    vs do mix de risco (risk-driven).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ResultadoPortfolio:
    n_vidas:           int
    premio_total:      float
    custo_esperado:    float
    loss_ratio:        float
    loss_ratio_target: float
    margem_tecnica:    float       # 1 - LR
    cv_lr:             float       # variância do LR via bootstrap
    p5_lr:             float
    p95_lr:            float


def simular_portfolio(
    glm_fit,                       # statsmodels GLMResults
    construir_features_func,        # callable para gerar X a partir do dict
    loss_ratio_target: float = 0.70,
    n_vidas: int = 10_000,
    seed: int = 42,
) -> tuple[pd.DataFrame, ResultadoPortfolio]:
    """Gera portfólio sintético e calcula prêmio + custo esperado."""
    rng = np.random.default_rng(seed)

    # Distribuição demográfica realística (US adult population)
    df = pd.DataFrame({
        "age":       rng.integers(18, 65, n_vidas),
        "bmi":       rng.normal(28.5, 6.2, n_vidas).clip(15, 55),
        "children":  rng.poisson(1.1, n_vidas).clip(0, 5),
        "smoker":    rng.choice(["no", "yes"], n_vidas, p=[0.81, 0.19]),
        "sex":       rng.choice(["male", "female"], n_vidas),
        "region":    rng.choice(["northeast", "northwest", "southeast", "southwest"], n_vidas),
    })

    # Features para o GLM (já com one-hot encoding)
    X = construir_features_func(df)
    df["custo_esperado"] = glm_fit.predict(X).values

    # Prêmio puro = custo esperado / loss ratio target
    df["premio_puro"] = df["custo_esperado"] / loss_ratio_target

    premio_total = float(df["premio_puro"].sum())
    custo_total = float(df["custo_esperado"].sum())
    lr = custo_total / premio_total

    # Bootstrap CV do LR (10k re-amostragens)
    lrs_boot = np.zeros(2000)
    n = len(df)
    custos = df["custo_esperado"].values
    premios = df["premio_puro"].values
    for i in range(2000):
        idx = rng.integers(0, n, n)
        lrs_boot[i] = custos[idx].sum() / premios[idx].sum()

    return df, ResultadoPortfolio(
        n_vidas=n_vidas,
        premio_total=premio_total,
        custo_esperado=custo_total,
        loss_ratio=lr,
        loss_ratio_target=loss_ratio_target,
        margem_tecnica=1.0 - lr,
        cv_lr=float(lrs_boot.std() / lrs_boot.mean()),
        p5_lr=float(np.percentile(lrs_boot, 5)),
        p95_lr=float(np.percentile(lrs_boot, 95)),
    )
