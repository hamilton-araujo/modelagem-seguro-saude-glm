"""Testes do GLM Gamma — diagnósticos e ratebook."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ingest import carregar
from features import construir_features, targets
from model import ajustar, prever, validar_diagnostico
from ratebook import construir, calcular_premio


@pytest.fixture(scope="module")
def dados():
    return carregar()


@pytest.fixture(scope="module")
def modelo(dados):
    X = construir_features(dados)
    y = targets(dados)
    return ajustar(X, y), X, y


class TestGLM:
    def test_pseudo_r2_positivo(self, modelo):
        res, _, _ = modelo
        assert res.pseudo_r2 > 0, f"Pseudo-R² negativo: {res.pseudo_r2:.4f}"

    def test_aic_finito(self, modelo):
        res, _, _ = modelo
        assert np.isfinite(res.aic)

    def test_n_correto(self, modelo, dados):
        res, _, _ = modelo
        assert res.n == len(dados)

    def test_smoker_coef_positivo(self, modelo):
        """Fumantes devem ter coeficiente positivo (maior custo)."""
        res, _, _ = modelo
        coef_smoker = res.coeficientes.loc["smoker_yes", "coeficiente"]
        assert coef_smoker > 0, f"Coef smoker_yes negativo: {coef_smoker:.4f}"

    def test_age_coef_positivo(self, modelo):
        """Idade deve elevar o custo."""
        res, _, _ = modelo
        assert res.coeficientes.loc["age", "coeficiente"] > 0

    def test_bmi_coef_positivo(self, modelo):
        """IMC deve elevar o custo."""
        res, _, _ = modelo
        assert res.coeficientes.loc["bmi", "coeficiente"] > 0

    def test_previsoes_positivas(self, modelo):
        res, X, _ = modelo
        y_pred = prever(res, X)
        assert (y_pred > 0).all(), "Todas as previsões devem ser positivas"

    def test_previsoes_mesma_dimensao(self, modelo):
        res, X, y = modelo
        y_pred = prever(res, X)
        assert len(y_pred) == len(y)

    def test_diagnostico_aprovado(self, modelo):
        res, _, _ = modelo
        checks = validar_diagnostico(res)
        for chave, ok in checks.items():
            assert ok, f"Diagnóstico falhou: {chave}"

    def test_residuos_sem_nan(self, modelo):
        res, _, _ = modelo
        assert res.residuos_deviance.isna().sum() == 0


class TestRatebook:
    def test_base_rate_positivo(self, modelo):
        res, _, _ = modelo
        rb = construir(res)
        assert rb.base_rate > 0

    def test_smoker_relativity_maior_1(self, modelo):
        """Fumante deve ter relativity > 1 (agrava o risco)."""
        res, _, _ = modelo
        rb = construir(res)
        assert rb.relatividades.loc["smoker_yes", "relativity"] > 1.0

    def test_premio_fumante_maior_nao_fumante(self, modelo):
        res, _, _ = modelo
        rb = construir(res)
        p_nao_fumante = calcular_premio(rb, 35, 28.0, 1, smoker=False, sex_male=False)
        p_fumante     = calcular_premio(rb, 35, 28.0, 1, smoker=True,  sex_male=False)
        assert p_fumante > p_nao_fumante

    def test_premio_mais_velho_maior(self, modelo):
        res, _, _ = modelo
        rb = construir(res)
        p_jovem  = calcular_premio(rb, 25, 25.0, 0, smoker=False, sex_male=False)
        p_velho  = calcular_premio(rb, 55, 25.0, 0, smoker=False, sex_male=False)
        assert p_velho > p_jovem

    def test_premio_positivo_qualquer_perfil(self, modelo):
        res, _, _ = modelo
        rb = construir(res)
        for region in ["northeast", "northwest", "southeast", "southwest"]:
            p = calcular_premio(rb, 30, 22.0, 0, smoker=False, sex_male=True, region=region)
            assert p > 0

    def test_coeficientes_exportaveis(self, modelo, tmp_path):
        res, _, _ = modelo
        path = tmp_path / "coefs.csv"
        res.coeficientes.to_csv(path)
        df = pd.read_csv(path, index_col=0)
        assert "coeficiente" in df.columns
        assert len(df) > 0
