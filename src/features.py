"""
Engenharia de features para o GLM.

Pipeline:
    1. One-Hot Encoding com drop da categoria de referência (evita multicolinearidade)
    2. Variáveis contínuas mantidas na escala original (GLM log-link lida nativamente)
    3. Matriz de design final pronta para statsmodels
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Categorias de referência (base level) — dropped para evitar dummy trap
REF_SEX    = "female"
REF_SMOKER = "no"
REF_REGION = "northeast"


def construir_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constrói a matriz de design X para o GLM.

    Variáveis contínuas: age, bmi, children (sem escalonamento — GLM log-link)
    Dummies: sex, smoker, region (com categoria de referência omitida)

    Returns:
        DataFrame com constante ('const') + features para statsmodels.
    """
    df = df.copy()

    # Dummies — omitir categoria de referência (full rank)
    df["smoker_yes"] = (df["smoker"] == "yes").astype("int8")
    df["sex_male"]   = (df["sex"] == "male").astype("int8")

    regioes = [r for r in df["region"].unique() if r != REF_REGION]
    for r in sorted(regioes):
        df[f"region_{r.replace(' ', '_')}"] = (df["region"] == r).astype("int8")

    # Constante para statsmodels
    df["const"] = 1.0

    feature_cols = (
        ["const", "age", "bmi", "children", "smoker_yes", "sex_male"]
        + sorted([c for c in df.columns if c.startswith("region_")])
    )

    X = df[feature_cols].astype("float32")
    logger.info("Matriz de design: %d × %d", *X.shape)
    return X


def targets(df: pd.DataFrame) -> pd.Series:
    """Retorna a variável alvo (charges) como float64 para statsmodels."""
    return df["charges"].astype("float64")


def descricao_features() -> dict[str, str]:
    """Dicionário com descrição legível de cada feature."""
    return {
        "const":        "Intercepto (perfil base)",
        "age":          "Idade (anos)",
        "bmi":          "IMC",
        "children":     "Número de dependentes",
        "smoker_yes":   "Fumante (vs não-fumante)",
        "sex_male":     "Sexo masculino (vs feminino)",
        "region_northwest": "Região Noroeste (vs Nordeste)",
        "region_southeast": "Região Sudeste (vs Nordeste)",
        "region_southwest": "Região Sudoeste (vs Nordeste)",
    }
