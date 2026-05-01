"""
Ratebook multiplicativo derivado do GLM.

Translada coeficientes logarítmicos em relatividades de subscrição:
    Relativity(X) = exp(β_X)
    Premio(perfil) = Base_Rate × ∏ Relativity(fator_i)
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from model import ResultadoGLM
from features import descricao_features

logger = logging.getLogger(__name__)


@dataclass
class Ratebook:
    """Tabela de relatividades e calculadora de prêmio por perfil."""
    base_rate:     float           # prêmio do perfil padrão (todos dummies = 0)
    relatividades: pd.DataFrame    # fator, relativity, IC_inf, IC_sup, p_valor


def construir(resultado: ResultadoGLM) -> Ratebook:
    """
    Constrói o ratebook a partir dos coeficientes do GLM.

    O perfil base é: female, não-fumante, northeast.
    Base rate = exp(β_intercepto).
    """
    coefs = resultado.coeficientes
    descricoes = descricao_features()

    base_rate = float(np.exp(coefs.loc["const", "coeficiente"]))

    rows = []
    for feat, row in coefs.iterrows():
        if feat == "const":
            continue
        rows.append({
            "feature":     feat,
            "descricao":   descricoes.get(feat, feat),
            "coeficiente": row["coeficiente"],
            "relativity":  row["relativity"],
            "ic_inf_95":   float(np.exp(row["ic_inf_95"])),
            "ic_sup_95":   float(np.exp(row["ic_sup_95"])),
            "p_valor":     row["p_valor"],
            "significante": row["p_valor"] < 0.05,
        })

    rel_df = pd.DataFrame(rows).set_index("feature")
    rel_df = rel_df.sort_values("relativity", ascending=False)

    logger.info("Ratebook construído — Base rate: $%.2f", base_rate)
    return Ratebook(base_rate=base_rate, relatividades=rel_df)


def calcular_premio(
    ratebook: Ratebook,
    age: int,
    bmi: float,
    children: int,
    smoker: bool,
    sex_male: bool,
    region: str = "northeast",
) -> float:
    """
    Calcula o prêmio para um perfil específico.

    Premio = Base_Rate × exp(β_age·age + β_bmi·bmi + ...)
    """
    rel = ratebook.relatividades["coeficiente"]

    premio = ratebook.base_rate
    premio *= np.exp(rel.get("age", 0) * age)
    premio *= np.exp(rel.get("bmi", 0) * bmi)
    premio *= np.exp(rel.get("children", 0) * children)

    if smoker:
        premio *= np.exp(rel.get("smoker_yes", 0))
    if sex_male:
        premio *= np.exp(rel.get("sex_male", 0))

    reg_key = f"region_{region.lower().replace(' ', '_')}"
    if reg_key in rel.index:
        premio *= np.exp(rel[reg_key])

    return float(premio)


def exportar_csv(ratebook: Ratebook, caminho) -> None:
    ratebook.relatividades.to_csv(caminho)
    logger.info("Ratebook exportado: %s", caminho)
