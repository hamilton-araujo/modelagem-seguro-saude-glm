"""
CLI — Tarifação de Seguro de Saúde via GLM Gamma.

Exemplos:
    python src/main.py
    python src/main.py --perfil 35 28.5 1 yes male southwest
    python src/main.py --no-report
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GLM Gamma para precificacao de seguro de saude.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--perfil", nargs=6,
        metavar=("AGE", "BMI", "CHILDREN", "SMOKER", "SEX", "REGION"),
        default=None,
        help="Calcular prêmio para perfil: ex: 35 28.5 1 yes male southwest",
    )
    p.add_argument("--no-report", action="store_true",
                   help="Suprimir gráficos e painel.")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def _validar_perfil(args) -> dict | None:
    if args.perfil is None:
        return None
    age, bmi, children, smoker, sex, region = args.perfil
    try:
        perfil = {
            "age":      int(age),
            "bmi":      float(bmi),
            "children": int(children),
            "smoker":   smoker.lower() == "yes",
            "sex_male": sex.lower() == "male",
            "region":   region.lower(),
        }
    except ValueError as e:
        print(f"Erro no perfil: {e}", file=sys.stderr)
        sys.exit(1)

    if not 18 <= perfil["age"] <= 64:
        print("Erro: --age deve estar entre 18 e 64.", file=sys.stderr)
        sys.exit(1)
    if perfil["bmi"] <= 0:
        print("Erro: --bmi deve ser positivo.", file=sys.stderr)
        sys.exit(1)
    regioes_validas = {"northeast", "northwest", "southeast", "southwest"}
    if perfil["region"] not in regioes_validas:
        print(f"Erro: região inválida. Válidas: {regioes_validas}", file=sys.stderr)
        sys.exit(1)
    return perfil


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    perfil = _validar_perfil(args)

    import ingest
    import features as feat
    import model as mdl
    import ratebook as rb_mod
    import report

    from pathlib import Path
    OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"\n{'━'*54}")
    print(f"  GLM Gamma — Precificacao de Seguro de Saude")
    print(f"{'━'*54}")

    print("\n[1/4] Carregando dataset Kaggle...")
    df = ingest.carregar()
    ingest.resumo(df)

    print("[2/4] Construindo features e ajustando GLM...")
    X = feat.construir_features(df)
    y = feat.targets(df)
    resultado = mdl.ajustar(X, y)

    print("[3/4] Construindo ratebook...")
    rb = rb_mod.construir(resultado)

    print("[4/4] Gerando relatorio...")
    y_pred = mdl.prever(resultado, X)

    if not args.no_report:
        report.gerar_relatorio_completo(resultado, rb, y, y_pred)
    else:
        report.exibir_painel(resultado, rb)

    # Exportar
    rb_mod.exportar_csv(rb, OUTPUT_DIR / "ratebook.csv")
    resultado.coeficientes.to_csv(OUTPUT_DIR / "coeficientes_glm.csv")
    print(f"[OK] Ratebook e coeficientes exportados para output/")

    # Calcular prêmio para perfil personalizado
    if perfil:
        premio = rb_mod.calcular_premio(rb, **perfil)
        print(f"\n  Premio para o perfil informado: ${premio:,.2f}\n")


if __name__ == "__main__":
    main()
