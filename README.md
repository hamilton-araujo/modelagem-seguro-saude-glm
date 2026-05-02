# Tarifação GLM Seguro Saúde — Decisão Executiva

> **A pergunta do CFO:** *Lançamos o produto novo? O ratebook discrimina bem risco? O Loss Ratio target de 70% é atingível em produção?*

Motor atuarial completo para tarifação de seguro de saúde via GLM, com **três camadas de validação executiva**: comparação multi-modelo, lift/Gini de discriminação atuarial e simulação de portfólio com Loss Ratio bootstrapped. Termina com decisão **LANÇAR / REVISAR / DESCONTINUAR**.

---

## Por que existe

O GLM Gamma tradicional entrega coeficientes e pseudo-R². Insuficiente: o CFO toma decisão com três incertezas:

| Incerteza | Sinal técnico |
|---|---|
| Escolhi a família certa? | Comparação Gamma vs Tweedie vs OLS log (AIC + RMSE OOS) |
| O ratebook discrimina? | Lift por decil + Gini coefficient |
| Loss Ratio em produção? | Simulação de 10.000 vidas + Bootstrap CI |

Este projeto consolida tudo em `output/relatorio_cfo.md` com decisão final.

---

## A história em três atos

### Ato 1 — A reunião de produto
Comitê na sexta. O time de produto pediu lançar plano novo com tarifa multiplicativa. Você roda:
```bash
python -m src.exec_report
```

### Ato 2 — A evidência
30 segundos depois:
```
Modelos comparados      3
Melhor AIC              OLS log(y)
Gini coefficient        0.405
Loss Ratio esperado     70.0%  (target 70%)
Margem técnica          30.0%
```

### Ato 3 — A decisão
```
✅ DECISÃO: LANÇAR
   · Loss Ratio P95 = 70.0% ≤ 75% (target + 5pp)
   · Gini = 0.405 ≥ 0.30 (boa discriminação atuarial)
```

---

## Modelos

### GLM Gamma + log link
```
ln(E[charges]) = β₀ + β·age + β·bmi + β·children
               + β·smoker + β·sex + β·region (3 dummies)
```
- Variável alvo contínua, positiva, **assimétrica à direita** → Gamma natural
- Var(Y) ∝ μ² (heterocedasticidade estrutural)
- Link log → previsões positivas + efeitos multiplicativos (ratebook direto)

### Tweedie (var_power = 1.5)
Composto Poisson-Gamma. Aceita zeros (apólices sem sinistro). Generaliza Gamma para portfólios com sinistralidade parcial.

### OLS sobre log(charges)
Baseline. Ajusta no log mas viola homocedasticidade — útil como benchmark.

### Gini Coefficient — discriminação atuarial
```
Gini = 2 × (0.5 − Área sob a curva de Lorenz dos custos rankeados pelo prêmio)
```

| Gini | Qualidade |
|---|---|
| ≥ 0.50 | Excelente |
| 0.30–0.50 | Bom |
| 0.20–0.30 | Marginal |
| < 0.20 | Não discrimina |

### Loss Ratio bootstrapped
Gera portfólio sintético de 10.000 vidas com distribuição demográfica realística (US adult population), aplica o ratebook e mede:
```
Loss Ratio = Σ Custos / Σ Prêmios
```
Bootstrap 2.000 reamostragens → CI 90%.

---

## Decisão executiva

| Decisão | Critério |
|---|---|
| ✅ LANÇAR | LR P95 ≤ target+5pp **E** Gini ≥ 0.30 |
| ⚠️ REVISAR | LR P95 ∈ (target+5pp, +10pp] **OU** Gini ∈ [0.20, 0.30) |
| ❌ DESCONTINUAR | LR P95 > target+10pp **OU** Gini < 0.20 |

---

## Dataset

**Medical Cost Personal Dataset** (Kaggle `mirichoi0218/insurance`) — 1.338 segurados, 7 variáveis (age, sex, bmi, children, smoker, region, charges).

---

## Stack

| Camada | Tecnologia |
|---|---|
| Dados | Kaggle API · Pandas · NumPy |
| Modelagem | statsmodels (GLM Gamma + Tweedie) |
| Validação | scikit-learn KFold · Lift/Gini · Bootstrap |
| Visualização | matplotlib · seaborn |

---

## Como rodar

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Pipeline completo (recomendado)
python -m src.exec_report

# CLI tradicional (modelo + ratebook)
python -m src.main
python -m src.main --perfil 35 28.5 1 yes male southwest
```

---

## Outputs

```
output/
├── relatorio_cfo.md                # ⭐ Briefing CFO com decisão
├── modelos_comparacao.png          # ⭐ AIC + RMSE OOS por família
├── modelos_comparacao.csv
├── lift_curve.png                  # ⭐ Lift por decil
├── lift_decis.csv
├── loss_ratio_dist.png             # ⭐ Bootstrap LR portfólio 10k vidas
├── portfolio_amostra.csv
├── coeficientes_glm.csv
├── ratebook.csv                    # Ratebook multiplicativo
├── glm_relatividades.png
├── glm_real_vs_previsto.png
└── glm_residuos.png
```

⭐ = adicionado nesta versão.

---

## Estrutura

```
├── src/
│   ├── exec_report.py            # ⭐ Pipeline executivo
│   ├── model_comparison.py       # ⭐ Gamma vs Tweedie vs OLS
│   ├── lift_analysis.py          # ⭐ Lift + Gini
│   ├── portfolio_simulation.py   # ⭐ 10k vidas + bootstrap LR
│   ├── ingest.py
│   ├── features.py
│   ├── model.py                  # GLM Gamma + log link
│   ├── ratebook.py
│   ├── report.py
│   └── main.py
├── data/
├── output/
├── tests/
└── requirements.txt
```
