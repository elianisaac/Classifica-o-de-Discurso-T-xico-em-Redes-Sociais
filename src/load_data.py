"""
load_data.py
============

Carrega o dataset ToLD-BR diretamente do repositório oficial do GitHub
e converte para o formato Arrow do Hugging Face (data/processed/).
Isso evita o erro de scripts descontinuados na biblioteca `datasets` v3+.

Uso:
    python src/load_data.py --variant binary --output data/processed/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import ensure_dir, save_results, setup_logging

# URL do CSV original no GitHub oficial
GITHUB_CSV_URL = "https://raw.githubusercontent.com/JAugusto97/ToLD-BR/main/ToLD-BR.csv"
VALID_VARIANTS = ("binary", "multilabel")
VALID_CONFIGS = {
    "binary": "told-br-binary",
    "multilabel": "told-br-multilabel",
}


def load_variant(variant: str, output_dir: Path, force: bool = False):
    """
    Baixa o dataset do GitHub, faz a divisão 80/10/10 e salva em disco.
    """
    logger = setup_logging(name="load_data")

    if variant not in VALID_VARIANTS:
        raise ValueError(f"Variante inválida: {variant}. Use uma de {VALID_VARIANTS}")

    cache_path = output_dir / VALID_CONFIGS[variant]

    if cache_path.exists() and not force:
        logger.info("Carregando do disco: %s", cache_path)
        return load_from_disk(str(cache_path))

    logger.info("Baixando dados brutos do GitHub: %s", GITHUB_CSV_URL)
    df = pd.read_csv(GITHUB_CSV_URL)

    # Divisão Treino (80%), Validação (10%), Teste (10%)
    # Usamos stratify no binário para manter a proporção de tóxicos
    if variant == "binary":
        df = df[['text', 'toxic']].rename(columns={'toxic': 'label'})
        df['label'] = df['label'].fillna(0).astype(int)
        
        train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
    else:
        df = df.drop(columns=['toxic'])
        train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    logger.info("Tamanhos gerados -> Treino: %d | Validação: %d | Teste: %d", 
                len(train_df), len(val_df), len(test_df))

    # Convertendo para DatasetDict do Hugging Face
    ds = DatasetDict({
        'train': Dataset.from_pandas(train_df, preserve_index=False),
        'validation': Dataset.from_pandas(val_df, preserve_index=False),
        'test': Dataset.from_pandas(test_df, preserve_index=False)
    })

    ensure_dir(cache_path.parent)
    logger.info("Salvando em disco: %s", cache_path)
    ds.save_to_disk(str(cache_path))

    return ds


def main() -> None:
    parser = argparse.ArgumentParser(description="Carrega o ToLD-BR via GitHub.")
    parser.add_argument("--variant", type=str, default="binary", choices=list(VALID_VARIANTS) + ["both"])
    parser.add_argument("--output", type=str, default="data/processed")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logger = setup_logging(log_file="results/logs/load_data.log", name="load_data")
    output_dir = Path(args.output)
    ensure_dir(output_dir)

    variants = VALID_VARIANTS if args.variant == "both" else (args.variant,)

    for variant in variants:
        logger.info("Processando variante: %s", variant)
        load_variant(variant, output_dir, force=args.force)
    
    logger.info("Concluído com sucesso.")


if __name__ == "__main__":
    main()