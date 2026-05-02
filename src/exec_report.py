"""
Relatório executivo do CFO — Tarifação GLM Seguro Saúde.

Por que existe:
    O CFO toma uma decisão por trimestre: lançar produto novo, revisar
    ratebook ou descontinuar linha. Este pipeline consolida em 1 markdown
    a evidência atuarial: comparação multi-modelo + lift/Gini + portfólio
    simulado + decisão final.

Decisão:
    LANÇAR     : LR p95 ≤ target+5pp  E  Gini ≥ 0.30
    REVISAR    : LR p95 ∈ (target+5pp, target+10pp]  OU  Gini ∈ [0.20, 0.30)
    DESCONTINUAR: LR p95 > target+10pp  OU  Gini < 0.20
"""

import io
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ingest
import features as feat
import model as mdl
from src.model_comparison import comparar
from src.lift_analysis import lift_decis, gini_coefficient
from src.portfolio_simulation import simular_portfolio

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _decisao(lr_p95: float, lr_target: float, gini: float) -> tuple[str, list[str]]:
    razoes = []
    delta = lr_p95 - lr_target
    if delta > 0.10:
        razoes.append(f"Loss Ratio P95 = {lr_p95*100:.1f}% (target {lr_target*100:.0f}% + 10pp = {(lr_target+0.10)*100:.0f}% violado)")
        return "DESCONTINUAR", razoes
    if gini < 0.20:
        razoes.append(f"Gini = {gini:.3f} < 0.20 — modelo não discrimina risco")
        return "DESCONTINUAR", razoes
    if delta <= 0.05 and gini >= 0.30:
        razoes.append(f"Loss Ratio P95 = {lr_p95*100:.1f}% ≤ {(lr_target+0.05)*100:.0f}% (target + 5pp)")
        razoes.append(f"Gini = {gini:.3f} ≥ 0.30 (boa discriminação atuarial)")
        return "LANCAR", razoes
    razoes.append(f"Loss Ratio P95 = {lr_p95*100:.1f}% · Gini = {gini:.3f}")
    return "REVISAR", razoes


def _grafico_comparacao(df_cmp: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    df_ok = df_cmp[df_cmp["converged"]]

    axes[0].barh(df_ok["nome"], df_ok["aic"], color="#3498db", alpha=0.85)
    axes[0].set_xlabel("AIC (menor = melhor)")
    axes[0].set_title("Comparação por AIC")
    axes[0].grid(alpha=0.3, axis="x")

    axes[1].barh(df_ok["nome"], df_ok["rmse_oos"], color="#e67e22", alpha=0.85)
    axes[1].set_xlabel("RMSE Out-of-Sample (USD)")
    axes[1].set_title("Erro de Predição (5-Fold CV)")
    axes[1].grid(alpha=0.3, axis="x")

    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _grafico_lift(df_lift: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(df_lift["decil"], df_lift["lift"], color="#9b59b6", alpha=0.85)
    ax.axhline(1.0, color="black", lw=1, ls="--", label="Custo médio global")
    for bar, l in zip(bars, df_lift["lift"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{l:.2f}×", ha="center", fontsize=9)
    ax.set_xlabel("Decil de prêmio previsto (1 = mais barato → 10 = mais caro)")
    ax.set_ylabel("Lift = custo real médio / custo médio global")
    ax.set_title("Lift Curve por Decil — Discriminação do Ratebook")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _grafico_loss_ratio(res, lrs_dist: np.ndarray, out: Path):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.hist(lrs_dist * 100, bins=60, color="#3498db", alpha=0.75)
    ax.axvline(res.loss_ratio * 100, color="red", lw=2,
               label=f"LR observado = {res.loss_ratio*100:.1f}%")
    ax.axvline(res.loss_ratio_target * 100, color="green", lw=2, ls="--",
               label=f"Target = {res.loss_ratio_target*100:.0f}%")
    ax.axvline(res.p95_lr * 100, color="orange", lw=1.2, ls=":",
               label=f"P95 = {res.p95_lr*100:.1f}%")
    ax.set_xlabel("Loss Ratio (%)")
    ax.set_ylabel("Frequência")
    ax.set_title(f"Distribuição Bootstrap — Portfólio {res.n_vidas:,} vidas")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _gerar_markdown(df_cmp, df_lift, gini, res, decisao, razoes, out: Path):
    badge = {"LANCAR": "✅", "REVISAR": "⚠️", "DESCONTINUAR": "❌"}[decisao]
    lines = [
        "# Tarifação Seguro Saúde — Decisão CFO",
        "",
        f"## Recomendação: {badge} **{decisao.replace('_', ' ').title()}**",
        "",
    ]
    for r in razoes:
        lines.append(f"- {r}")

    lines += [
        "",
        "---",
        "",
        "## 1. Comparação Multi-Modelo",
        "",
        "| Modelo | AIC | RMSE OOS | Pseudo-R² |",
        "|---|---|---|---|",
    ]
    for _, r in df_cmp.iterrows():
        lines.append(f"| {r['nome']} | {r['aic']:.0f} | {r['rmse_oos']:.0f} | {r['pseudo_r2']:.3f} |")

    lines += [
        "",
        "![Comparação](modelos_comparacao.png)",
        "",
        "## 2. Discriminação Atuarial (Lift + Gini)",
        "",
        f"- **Gini coefficient:** {gini:.3f} *(≥ 0.30 = bom modelo, ≥ 0.50 = excelente)*",
        f"- **Spread Decil 10 / Decil 1:** {df_lift.iloc[-1]['lift'] / df_lift.iloc[0]['lift']:.1f}×",
        "",
        "![Lift](lift_curve.png)",
        "",
        "| Decil | n | Custo Real Médio | Prêmio Médio | Lift | LR Relativo |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in df_lift.iterrows():
        lines.append(f"| {int(r['decil'])} | {int(r['n'])} | "
                     f"${r['custo_real_medio']:,.0f} | "
                     f"${r['premio_pred_medio']:,.0f} | "
                     f"{r['lift']:.2f}× | {r['loss_ratio_relativo']:.2f} |")

    lines += [
        "",
        "## 3. Simulação de Portfólio",
        "",
        f"- **Vidas simuladas:** {res.n_vidas:,}",
        f"- **Prêmio total esperado:** ${res.premio_total/1e6:.2f}M",
        f"- **Custo esperado:** ${res.custo_esperado/1e6:.2f}M",
        f"- **Loss Ratio esperado:** {res.loss_ratio*100:.1f}% (target: {res.loss_ratio_target*100:.0f}%)",
        f"- **Margem técnica:** {res.margem_tecnica*100:.1f}%",
        f"- **CI 90% bootstrap:** [{res.p5_lr*100:.1f}%, {res.p95_lr*100:.1f}%]",
        "",
        "![Loss Ratio](loss_ratio_dist.png)",
        "",
        "---",
        "",
        "## Critérios da decisão",
        "",
        "| Decisão | Critério |",
        "|---|---|",
        "| ✅ LANÇAR | LR P95 ≤ target+5pp **E** Gini ≥ 0.30 |",
        "| ⚠️ REVISAR | LR P95 ∈ (target+5pp, +10pp] **OU** Gini ∈ [0.20, 0.30) |",
        "| ❌ DESCONTINUAR | LR P95 > target+10pp **OU** Gini < 0.20 |",
        "",
        "## Metodologia",
        "",
        "- **GLM Gamma + log link**: Var(Y) ∝ μ², adequado para custos assimétricos.",
        "- **Tweedie (var_power=1.5)**: composto Poisson-Gamma, aceita zeros.",
        "- **Lift por decil**: divide base em 10 grupos por prêmio previsto, mede custo real médio relativo.",
        "- **Gini coefficient**: 2×(0.5 − área sob curva de Lorenz dos custos rankeados pelo prêmio).",
        "- **Bootstrap LR**: 2.000 re-amostragens com reposição → CI 90%.",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. Dados
    df = ingest.carregar()
    X = feat.construir_features(df)
    y = feat.targets(df)

    # 2. GLM Gamma final (modelo principal)
    glm_res = mdl.ajustar(X, y)

    # 3. Comparação multi-modelo
    df_cmp = comparar(X, y, k_folds=5)
    df_cmp.to_csv(OUTPUT_DIR / "modelos_comparacao.csv", index=False)

    # 4. Lift + Gini sobre os dados in-sample
    y_pred = glm_res.fit.predict(X)
    df_lift = lift_decis(y, y_pred)
    df_lift.to_csv(OUTPUT_DIR / "lift_decis.csv", index=False)
    gini = gini_coefficient(y.values, y_pred.values)

    # 5. Simulação de portfólio
    portfolio_df, res = simular_portfolio(
        glm_res.fit,
        feat.construir_features,
        loss_ratio_target=0.70, n_vidas=10_000,
    )
    portfolio_df.head(100).to_csv(OUTPUT_DIR / "portfolio_amostra.csv", index=False)

    # Bootstrap distribution para gráfico
    rng = np.random.default_rng(42)
    custos = portfolio_df["custo_esperado"].values
    premios = portfolio_df["premio_puro"].values
    n = len(portfolio_df)
    lrs_dist = np.array([
        custos[rng.integers(0, n, n)].sum() / premios[rng.integers(0, n, n)].sum()
        for _ in range(2000)
    ])

    # 6. Decisão
    decisao, razoes = _decisao(res.p95_lr, res.loss_ratio_target, gini)

    # 7. Charts + markdown
    _grafico_comparacao(df_cmp, OUTPUT_DIR / "modelos_comparacao.png")
    _grafico_lift(df_lift, OUTPUT_DIR / "lift_curve.png")
    _grafico_loss_ratio(res, lrs_dist, OUTPUT_DIR / "loss_ratio_dist.png")
    _gerar_markdown(df_cmp, df_lift, gini, res, decisao, razoes,
                    OUTPUT_DIR / "relatorio_cfo.md")

    badge = {"LANCAR": "✅", "REVISAR": "⚠️", "DESCONTINUAR": "❌"}[decisao]
    print(f"\n{'═'*60}")
    print(f"  TARIFAÇÃO GLM SEGURO SAÚDE — DECISÃO CFO")
    print(f"{'═'*60}")
    print(f"  Modelos comparados      {len(df_cmp)}")
    print(f"  Melhor AIC              {df_cmp.loc[df_cmp['aic'].idxmin(), 'nome']} ({df_cmp['aic'].min():.0f})")
    print(f"  Gini coefficient        {gini:.3f}")
    print(f"  Loss Ratio esperado     {res.loss_ratio*100:.1f}%  (target {res.loss_ratio_target*100:.0f}%)")
    print(f"  CI 90% LR               [{res.p5_lr*100:.1f}%, {res.p95_lr*100:.1f}%]")
    print(f"  Margem técnica          {res.margem_tecnica*100:.1f}%")
    print(f"{'═'*60}")
    print(f"  {badge}  DECISÃO: {decisao.replace('_', ' ').title()}")
    for r in razoes:
        print(f"     · {r}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
