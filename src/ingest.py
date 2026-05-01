"""Carga e validação do dataset Medical Cost Personal (Kaggle)."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_PATH = DATA_DIR / "insurance.csv"

COLUNAS_ESPERADAS = {"age", "sex", "bmi", "children", "smoker", "region", "charges"}


def carregar() -> pd.DataFrame:
    """
    Carrega o dataset insurance.csv e valida integridade básica.

    Returns:
        DataFrame limpo com tipos corretos e sem nulos.
    """
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado em {CSV_PATH}.\n"
            "Execute: kaggle datasets download -d mirichoi0218/insurance"
        )

    df = pd.read_csv(CSV_PATH)

    ausentes = COLUNAS_ESPERADAS - set(df.columns)
    if ausentes:
        raise ValueError(f"Colunas ausentes no dataset: {ausentes}")

    df = _limpar(df)
    logger.info("Dataset carregado: %d linhas × %d colunas", *df.shape)
    return df


def _limpar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Tipos
    df["age"]      = df["age"].astype("int16")
    df["children"] = df["children"].astype("int8")
    df["bmi"]      = df["bmi"].astype("float32")
    df["charges"]  = df["charges"].astype("float32")

    # Normalizar categóricas
    for col in ["sex", "smoker", "region"]:
        df[col] = df[col].str.strip().str.lower()

    # Remover nulos e negativos
    df = df.dropna()
    df = df[df["charges"] > 0]
    df = df[df["bmi"] > 0]

    logger.info("Após limpeza: %d linhas", len(df))
    return df.reset_index(drop=True)


def resumo(df: pd.DataFrame) -> None:
    print(f"\n{'─'*52}")
    print(f"  Medical Cost Dataset — {len(df):,} segurados")
    print(f"{'─'*52}")
    print(f"  Charges — Média: ${df['charges'].mean():,.0f}  "
          f"Mediana: ${df['charges'].median():,.0f}  "
          f"Máx: ${df['charges'].max():,.0f}")
    print(f"  Fumantes : {(df['smoker']=='yes').mean():.1%}")
    print(f"  IMC médio: {df['bmi'].mean():.1f}")
    print(f"  Idade    : {df['age'].min()}–{df['age'].max()} anos")
    print(f"{'─'*52}\n")
