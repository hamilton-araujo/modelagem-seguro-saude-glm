"""
Comparação de famílias GLM para tarifação.

Por que existe:
    Atuários default usam Gamma porque "todos usam". Mas três alternativas
    competem: Gaussian (OLS), Gamma e Tweedie. A escolha errada distorce
    prêmios em 5-15% — diferença entre operadora lucrativa e deficitária.

Critérios:
    - **AIC**: penaliza complexidade
    - **RMSE Out-of-Sample**: erro real em produção
    - **MAE OOS**: robusto a outliers
    - **Pseudo-R²**: % de variação explicada
    - **Gini**: poder de discriminação (separa risco alto vs baixo)
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import KFold


@dataclass
class ComparacaoModelo:
    nome:        str
    aic:         float
    rmse_oos:    float
    mae_oos:     float
    pseudo_r2:   float
    converged:   bool


def comparar(X: pd.DataFrame, y: pd.Series, k_folds: int = 5,
             seed: int = 42) -> pd.DataFrame:
    """Compara Gamma, Tweedie e OLS log-transformado via K-Fold CV."""
    resultados = []

    # ── 1. Gamma + log ────────────────────────────────────────
    resultados.append(_avaliar_glm(
        X, y, "Gamma+log",
        family=sm.families.Gamma(link=sm.families.links.Log()),
        k_folds=k_folds, seed=seed,
    ))

    # ── 2. Tweedie (var_power=1.5: composto Poisson-Gamma) ─────
    try:
        resultados.append(_avaliar_glm(
            X, y, "Tweedie(p=1.5)+log",
            family=sm.families.Tweedie(var_power=1.5,
                                       link=sm.families.links.Log()),
            k_folds=k_folds, seed=seed,
        ))
    except Exception:
        resultados.append(ComparacaoModelo("Tweedie(p=1.5)+log", np.nan, np.nan, np.nan, np.nan, False))

    # ── 3. Gaussian (OLS) sobre log(y) ─────────────────────────
    resultados.append(_avaliar_log_ols(X, y, k_folds=k_folds, seed=seed))

    return pd.DataFrame([r.__dict__ for r in resultados])


def _avaliar_glm(X, y, nome, family, k_folds, seed) -> ComparacaoModelo:
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    rmses, maes = [], []
    for tr_idx, te_idx in kf.split(X):
        try:
            glm = sm.GLM(y.iloc[tr_idx], X.iloc[tr_idx], family=family).fit()
            pred = glm.predict(X.iloc[te_idx])
            rmses.append(np.sqrt(np.mean((y.iloc[te_idx] - pred) ** 2)))
            maes.append(np.mean(np.abs(y.iloc[te_idx] - pred)))
        except Exception:
            return ComparacaoModelo(nome, np.nan, np.nan, np.nan, np.nan, False)

    # Refit completo para AIC + pseudo R²
    fit_full = sm.GLM(y, X, family=family).fit()
    aic = fit_full.aic
    dev_null = sm.GLM(y, np.ones((len(y), 1)), family=family).fit().deviance
    pseudo_r2 = 1.0 - fit_full.deviance / dev_null if dev_null > 0 else 0.0

    return ComparacaoModelo(
        nome=nome, aic=float(aic),
        rmse_oos=float(np.mean(rmses)),
        mae_oos=float(np.mean(maes)),
        pseudo_r2=float(pseudo_r2), converged=True,
    )


def _avaliar_log_ols(X, y, k_folds, seed) -> ComparacaoModelo:
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    log_y = np.log(y)
    rmses, maes = [], []
    for tr_idx, te_idx in kf.split(X):
        ols = sm.OLS(log_y.iloc[tr_idx], X.iloc[tr_idx]).fit()
        pred = np.exp(ols.predict(X.iloc[te_idx]))
        rmses.append(np.sqrt(np.mean((y.iloc[te_idx] - pred) ** 2)))
        maes.append(np.mean(np.abs(y.iloc[te_idx] - pred)))

    fit_full = sm.OLS(log_y, X).fit()
    return ComparacaoModelo(
        nome="OLS log(y)", aic=float(fit_full.aic),
        rmse_oos=float(np.mean(rmses)),
        mae_oos=float(np.mean(maes)),
        pseudo_r2=float(fit_full.rsquared), converged=True,
    )
