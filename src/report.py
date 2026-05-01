"""
Painel CLI e gráficos de diagnóstico do GLM.

Saídas:
    - Painel de texto no terminal
    - Gráfico de resíduos de deviance vs valores ajustados
    - Histograma de resíduos
    - Gráfico de relatividades (barras horizontais)
    - Gráfico real vs previsto
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

from model import ResultadoGLM
from ratebook import Ratebook

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

_DARK  = "#16213E"
_BLUE  = "#2E86AB"
_RED   = "#E84855"
_GREEN = "#52B788"
_AMBER = "#F4A261"
_SEP   = "=" * 54


def exibir_painel(res: ResultadoGLM, rb: Ratebook) -> None:
    """Imprime painel analítico no terminal."""
    print(f"\n{_SEP}")
    print(f"  GLM GAMMA — TARIFACAO SEGURO SAUDE")
    print(_SEP)
    print(f"  Observacoes    : {res.n:,}")
    print(f"  AIC            : {res.aic:,.1f}")
    print(f"  BIC            : {res.bic:,.1f}")
    print(f"  Deviance       : {res.deviance:,.1f}")
    print(f"  Pseudo-R2      : {res.pseudo_r2:.4f}")
    print(_SEP)
    print(f"  COEFICIENTES E RELATIVIDADES")
    print(f"  {'Feature':<24} {'Coef':>8} {'Relat.':>8} {'p-valor':>9}")
    print(f"  {'-'*24}  {'-'*8}  {'-'*8}  {'-'*9}")
    for feat, row in res.coeficientes.iterrows():
        sig = "*" if row["p_valor"] < 0.05 else " "
        print(f"  {feat:<24} {row['coeficiente']:>8.4f}  "
              f"{row['relativity']:>7.3f}x  {row['p_valor']:>9.4f}{sig}")
    print(_SEP)
    print(f"  RATEBOOK — PERFIL BASE: female, nao-fumante, NE")
    print(f"  Base Rate      : ${rb.base_rate:>10,.2f}")
    print(f"  Tabagismo (+)  : {rb.relatividades.loc['smoker_yes','relativity']:>9.3f}x")
    print(f"  Idade (+1 ano) : {rb.relatividades.loc['age','relativity']:>9.4f}x")
    print(f"  IMC (+1 unit.) : {rb.relatividades.loc['bmi','relativity']:>9.4f}x")
    print(f"{_SEP}\n")


def grafico_residuos(res: ResultadoGLM, prefixo: str = "glm") -> Path:
    """Resíduos de deviance vs valores ajustados."""
    fitted  = pd.Series(res.fit.fittedvalues)
    residuos = res.residuos_deviance

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(_DARK)

    # Resíduos vs ajustados
    ax = axes[0]
    ax.set_facecolor("#1A1A2E")
    ax.scatter(fitted, residuos, alpha=0.4, s=15, color=_BLUE)
    ax.axhline(0, color=_AMBER, lw=1.5, ls="--")
    ax.set_xlabel("Valores Ajustados (USD)", color="white")
    ax.set_ylabel("Resíduo de Deviance", color="white")
    ax.set_title("Resíduos vs Ajustados", color="white")
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333366")

    # Histograma de resíduos
    ax2 = axes[1]
    ax2.set_facecolor("#1A1A2E")
    sns.histplot(residuos, bins=40, color=_BLUE, ax=ax2, kde=True,
                 line_kws={"color": _AMBER, "lw": 2})
    ax2.axvline(0, color=_RED, lw=1.5, ls="--")
    ax2.set_xlabel("Resíduo de Deviance", color="white")
    ax2.set_ylabel("Frequência", color="white")
    ax2.set_title("Distribuição dos Resíduos", color="white")
    ax2.tick_params(colors="white")
    for sp in ax2.spines.values():
        sp.set_edgecolor("#333366")

    plt.tight_layout()
    path = OUTPUT_DIR / f"{prefixo}_residuos.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=_DARK)
    plt.close(fig)
    logger.info("Gráfico salvo: %s", path.name)
    return path


def grafico_relatividades(rb: Ratebook, prefixo: str = "glm") -> Path:
    """Barras horizontais das relatividades por feature (excl. contínuas)."""
    rel = rb.relatividades.copy()
    # Mostrar apenas dummies + relatividade expressiva
    rel = rel[rel["p_valor"] < 0.05].copy()
    rel = rel.sort_values("relativity")

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(_DARK)
    ax.set_facecolor("#1A1A2E")

    colors = [_RED if r > 1 else _GREEN for r in rel["relativity"]]
    bars = ax.barh(rel["descricao"], rel["relativity"], color=colors, alpha=0.85)
    ax.axvline(1.0, color=_AMBER, lw=1.5, ls="--", label="Sem efeito (1.0×)")

    for bar, (_, row) in zip(bars, rel.iterrows()):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{row['relativity']:.3f}×", va="center", color="white", fontsize=9)

    ax.set_xlabel("Relativity (exp(β))", color="white")
    ax.set_title("Relatividades Significativas — GLM Gamma", color="white")
    ax.tick_params(colors="white")
    ax.legend(framealpha=0.3, labelcolor="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333366")

    plt.tight_layout()
    path = OUTPUT_DIR / f"{prefixo}_relatividades.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=_DARK)
    plt.close(fig)
    logger.info("Gráfico salvo: %s", path.name)
    return path


def grafico_real_vs_previsto(y_real: pd.Series, y_pred: pd.Series,
                             prefixo: str = "glm") -> Path:
    """Scatter: charges reais vs charges previstos pelo GLM."""
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(_DARK)
    ax.set_facecolor("#1A1A2E")

    ax.scatter(y_real, y_pred, alpha=0.35, s=12, color=_BLUE)
    lim = max(y_real.max(), y_pred.max()) * 1.05
    ax.plot([0, lim], [0, lim], color=_AMBER, lw=1.5, ls="--", label="Ideal (y=x)")

    ax.set_xlabel("Charges Reais (USD)", color="white")
    ax.set_ylabel("Charges Previstos (USD)", color="white")
    ax.set_title("Real vs Previsto — GLM Gamma", color="white")
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.tick_params(colors="white")
    ax.legend(framealpha=0.3, labelcolor="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333366")

    plt.tight_layout()
    path = OUTPUT_DIR / f"{prefixo}_real_vs_previsto.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=_DARK)
    plt.close(fig)
    logger.info("Gráfico salvo: %s", path.name)
    return path


def gerar_relatorio_completo(res: ResultadoGLM, rb: Ratebook,
                             y_real: pd.Series, y_pred: pd.Series,
                             prefixo: str = "glm") -> None:
    exibir_painel(res, rb)
    grafico_residuos(res, prefixo)
    grafico_relatividades(rb, prefixo)
    grafico_real_vs_previsto(y_real, y_pred, prefixo)
    print(f"Gráficos salvos em output/\n")
