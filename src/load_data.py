"""
load_data.py
============

Carrega o dataset ToLD-BR do Hugging Face e o converte para disco local
(formato Arrow do `datasets`), evitando re-executar o loading script a cada
sessão do Colab.

Uso:
    python src/load_data.py --variant binary --output data/processed/
    python src/load_data.py --variant multilabel --output data/processed/
    python src/load_data.py --variant both --output data/processed/

Premissa do projeto: nenhum uso de Google Drive. A persistência é feita
recarregando do Hugging Face e/ou baixando os resultados ao final da sessão.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

# Permite importar utils.py quando o script é executado diretamente
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import (  # noqa: E402
    describe_environment,
    ensure_dir,
    save_results,
    setup_logging,
)


HF_DATASET = "JAugusto97/told-br"
VALID_VARIANTS = ("binary", "multilabel")
VALID_CONFIGS = {
    # Nome da config no HF : nome da pasta local
    "binary": "told-br-binary",
    "multilabel": "told-br-multilabel",
}


def load_variant(variant: str, output_dir: Path, force: bool = False):
    """
    Carrega uma variante do ToLD-BR do Hugging Face e salva em disco.

    Se a versão em disco já existir e `force=False`, apenas recarrega
    do disco (evita download e re-execução do loading script).

    Args:
        variant: 'binary' ou 'multilabel'.
        output_dir: diretório onde salvar/recarregar a versão em disco.
        force: se True, força o download e sobrescreve a versão em disco.

    Returns:
        DatasetDict do Hugging Face.
    """
    from datasets import load_dataset, load_from_disk

    logger = setup_logging(name="load_data")

    if variant not in VALID_VARIANTS:
        raise ValueError(f"Variante inválida: {variant}. Use uma de {VALID_VARIANTS}")

    cache_path = output_dir / VALID_CONFIGS[variant]

    if cache_path.exists() and not force:
        logger.info("Carregando do disco: %s", cache_path)
        return load_from_disk(str(cache_path))

    logger.info("Baixando do Hugging Face: %s (config=%s)", HF_DATASET, variant)

    ds = load_dataset(
        HF_DATASET,
        variant,
        trust_remote_code=True,  # o dataset usa loading script
    )

    ensure_dir(cache_path.parent)
    logger.info("Salvando em disco: %s", cache_path)
    ds.save_to_disk(str(cache_path))

    return ds


def describe_dataset(ds) -> dict:
    """Resume a composição do dataset para registro nos logs."""
    summary = {}
    for split_name, split in ds.items():
        summary[split_name] = {
            "num_examples": len(split),
            "features": list(split.features.keys()),
        }
        # Distribuição de classes, se houver coluna 'label' ou 'toxic'
        for label_col in ("label", "toxic"):
            if label_col in split.features:
                try:
                    counts = {}
                    for value in split[label_col]:
                        counts[str(value)] = counts.get(str(value), 0) + 1
                    summary[split_name][f"label_counts_{label_col}"] = counts
                except Exception:
                    pass
                break
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Carrega o ToLD-BR do Hugging Face e salva em disco."
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="binary",
        choices=list(VALID_VARIANTS) + ["both"],
        help="Variante a carregar (padrão: binary).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed",
        help="Diretório de saída (padrão: data/processed).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Força o download e sobrescreve a versão em disco.",
    )
    args = parser.parse_args()

    logger = setup_logging(log_file="results/logs/load_data.log", name="load_data")

    output_dir = Path(args.output)
    ensure_dir(output_dir)

    variants = VALID_VARIANTS if args.variant == "both" else (args.variant,)

    report = {"environment": describe_environment(), "datasets": {}}

    for variant in variants:
        logger.info("=" * 60)
        logger.info("Processando variante: %s", variant)
        logger.info("=" * 60)

        ds = load_variant(variant, output_dir, force=args.force)
        summary = describe_dataset(ds)

        logger.info("Resumo do dataset (%s):", variant)
        for split_name, info in summary.items():
            logger.info("  %s: %d exemplos | features=%s",
                        split_name, info["num_examples"], info["features"])
            for key, counts in info.items():
                if key.startswith("label_counts_"):
                    logger.info("    %s: %s", key, counts)

        report["datasets"][variant] = summary

    ensure_dir("results/metrics")
    save_results(report, "results/metrics", filename="dataset_summary.json")
    logger.info("Concluído.")


if __name__ == "__main__":
    main()