"""
GLM Gamma com link logarítmico para precificação de seguro de saúde.

Justificativa da família Gamma:
    - charges é contínuo, positivo e fortemente assimétrico à direita
    - Gamma assume Var(Y) = φ·μ² — adequado para custos de sinistro
    - Link log garante previsões positivas e efeitos multiplicativos interpretáveis

Otimização via MLE (statsmodels usa IRLS internamente).
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.families import Gamma
from statsmodels.genmod.families.links import Log

logger = logging.getLogger(__name__)


@dataclass
class ResultadoGLM:
    """Container com modelo ajustado e métricas de diagnóstico."""
    fit:               object           # statsmodels GLMResultsWrapper
    aic:               float
    bic:               float
    deviance:          float
    deviance_nula:     float
    pseudo_r2:         float            # McFadden: 1 - dev/dev_nula
    n:                 int
    coeficientes:      pd.DataFrame     # coef, std_err, z, p, IC 95%
    residuos_deviance: pd.Series


def ajustar(X: pd.DataFrame, y: pd.Series) -> ResultadoGLM:
    """
    Ajusta GLM Gamma com link log via MLE (IRLS).

    Args:
        X: Matriz de design com constante (saída de features.construir_features).
        y: Série de charges (float64, positivo).

    Returns:
        ResultadoGLM com modelo ajustado e diagnósticos.
    """
    familia = Gamma(link=Log())
    glm = sm.GLM(y, X, family=familia)
    fit = glm.fit()

    coefs = pd.DataFrame({
        "coeficiente":  fit.params,
        "std_err":      fit.bse,
        "z":            fit.tvalues,
        "p_valor":      fit.pvalues,
        "ic_inf_95":    fit.conf_int()[0],
        "ic_sup_95":    fit.conf_int()[1],
        "relativity":   np.exp(fit.params),
    })

    dev_nula = fit.null_deviance
    dev_mod  = fit.deviance
    pseudo_r2 = float(1 - dev_mod / dev_nula) if dev_nula > 0 else np.nan

    logger.info(
        "GLM Gamma ajustado — AIC: %.1f | Pseudo-R²: %.3f | Deviance: %.1f",
        fit.aic, pseudo_r2, dev_mod,
    )

    return ResultadoGLM(
        fit=fit,
        aic=float(fit.aic),
        bic=float(fit.bic),
        deviance=float(dev_mod),
        deviance_nula=float(dev_nula),
        pseudo_r2=pseudo_r2,
        n=int(fit.nobs),
        coeficientes=coefs,
        residuos_deviance=pd.Series(fit.resid_deviance, name="residuo_deviance"),
    )


def prever(resultado: ResultadoGLM, X: pd.DataFrame) -> pd.Series:
    """Retorna previsões do GLM na escala original (USD)."""
    return pd.Series(resultado.fit.predict(X), name="charges_previsto")


def validar_diagnostico(resultado: ResultadoGLM) -> dict[str, bool]:
    """
    Verifica premissas básicas do modelo.

    Returns:
        Dict com flags de aprovação para cada diagnóstico.
    """
    res = resultado.residuos_deviance

    checks = {
        "pseudo_r2_positivo":   resultado.pseudo_r2 > 0,
        "aic_finito":           np.isfinite(resultado.aic),
        "residuos_sem_nan":     res.isna().sum() == 0,
        "coefs_significativos": (resultado.coeficientes["p_valor"] < 0.05).any(),
        "smoker_positivo":      (
            "smoker_yes" in resultado.coeficientes.index and
            resultado.coeficientes.loc["smoker_yes", "coeficiente"] > 0
        ),
    }
    return checks
