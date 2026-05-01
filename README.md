# Modelagem Estatística para Tarifação de Seguro de Saúde (GLM)

Modelo atuarial de precificação utilizando Modelos Lineares Generalizados (GLMs) para isolar o risco marginal de fatores demográficos e biométricos, gerando um *ratebook* multiplicativo com interpretação direta para subscrição.

## Dataset

**Medical Cost Personal Dataset** — Kaggle (`mirichoi0218/insurance`)  
1.338 registros | 7 variáveis | Segurados de plano de saúde nos EUA

```
kaggle datasets download -d mirichoi0218/insurance
```

| Variável | Tipo | Descrição |
|---|---|---|
| age | int | Idade do segurado |
| sex | cat | Sexo (male/female) |
| bmi | float | Índice de Massa Corporal |
| children | int | Número de dependentes |
| smoker | cat | Tabagismo (yes/no) |
| region | cat | Região geográfica (4 zonas EUA) |
| charges | float | Custo anual do sinistro (USD) — variável alvo |

## Stack

| Camada | Tecnologia |
|---|---|
| Dados | Pandas · NumPy |
| Modelagem | statsmodels (GLM Gamma + log link) |
| Diagnóstico | AIC · Deviance Residuals · Pseudo-R² |
| Visualização | Matplotlib · Seaborn |

## Modelo

### Por que GLM Gamma?
- `charges` é **contínuo, positivo e assimétrico à direita** — viola premissas OLS
- Família **Gamma**: variância ∝ μ² (heterocedasticidade estrutural)
- **Link logarítmico**: garante previsões positivas e efeitos multiplicativos

```
ln(E[charges]) = β₀ + β₁·age + β₂·bmi + β₃·children
               + β₄·smoker_yes + β₅·sex_male
               + β₆·region_nw + β₇·region_se + β₈·region_sw
```

### Ratebook Multiplicativo
```
Relativity(X) = exp(β_X)
Prêmio(perfil) = Base_Rate × ∏ Relativity(fator_i)
```

## Saída

```
══════════════════════════════════════════════════
  GLM GAMMA — TARIFAÇÃO SEGURO SAÚDE
══════════════════════════════════════════════════
  AIC              :   19.842,3
  Pseudo-R² (McF.) :      0.312
  Deviance         :    352,4
══════════════════════════════════════════════════
  RATEBOOK MULTIPLICATIVO
  ──────────────────────────────────────────────
  Base Rate (perfil padrão)  :  $ 2.413,00
  Tabagismo (sim)            :     3.82×
  IMC (+1 unidade)           :     1.02×
  Idade (+1 ano)             :     1.03×
  Dependente (+1)            :     1.07×
══════════════════════════════════════════════════
```

## Estrutura

```
├── src/
│   ├── ingest.py      # Carga e validação do dataset Kaggle
│   ├── features.py    # Encoding WoE, dummies, escalonamento
│   ├── model.py       # GLM Gamma + diagnóstico AIC/deviance
│   ├── ratebook.py    # Relatividades e prêmio por perfil
│   ├── report.py      # Painel CLI + gráficos diagnósticos
│   └── main.py        # CLI argparse
├── data/
│   └── insurance.csv
├── output/
├── tests/
├── requirements.txt
└── README.md
```
