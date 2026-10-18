"""
preprocessing.py
================

Pré-processamento de texto para o dataset ToLD-BR.

Gera duas versões do texto:
1. `text_clean`: Limpeza agressiva (minúsculas, sem acentos, sem pontuação,
   sem stopwords) — ideal para a Regressão Logística com TF-IDF.
2. `text_bert`: Limpeza leve (apenas normaliza menções e URLs) — ideal para
   o BERTimbau, que lida com pontuação e contexto nativamente.

Uso:
    python src/preprocessing.py --input data/processed/told-br-binary --output data/processed/told-br-binary-clean
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import unidecode

# Permite importar utils.py quando o script é executado diretamente
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import (
    ensure_dir,
    load_config,
    save_results,
    setup_logging,
)

# ============================================================
# Listas de Stopwords (PT-BR)
# ============================================================
# Lista básica embutida para evitar dependência do NLTK no momento da limpeza.
# Você pode expandir isso carregando do NLTK se preferir.
STOPWORDS_PT = {
    "a", "à", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles", "aquilo",
    "as", "até", "com", "como", "da", "das", "de", "dela", "delas", "dele",
    "deles", "depois", "do", "dos", "e", "é", "ela", "elas", "ele", "eles",
    "em", "entre", "era", "eram", "éramos", "essa", "essas", "esse", "esses",
    "esta", "está", "estamos", "estão", "estar", "estas", "estava", "estavam",
    "estávamos", "este", "esteja", "estejam", "estejamos", "estes", "esteve",
    "estive", "estivemos", "estiver", "estivera", "estiveram", "estivéramos",
    "estiverem", "estivermos", "estivesse", "estivessem", "estivéssemos",
    "estou", "eu", "foi", "fomos", "for", "fora", "foram", "fôramos", "forem",
    "formos", "fosse", "fossem", "fôssemos", "fui", "há", "haja", "hajam",
    "hajamos", "hão", "havemos", "hei", "houve", "houvemos", "houver",
    "houvera", "houverá", "houveram", "houvéramos", "houverão", "houverei",
    "houverem", "houveremos", "houveria", "houveriam", "houveríamos",
    "houvermos", "houvesse", "houvessem", "houvéssemos", "isso", "isto",
    "já", "lhe", "lhes", "mais", "mas", "me", "mesmo", "meu", "meus", "minha",
    "minhas", "muito", "na", "não", "nas", "nem", "no", "nos", "nós", "nossa",
    "nossas", "nosso", "nossos", "num", "numa", "o", "os", "ou", "para", "pela",
    "pelas", "pelo", "pelos", "por", "qual", "quando", "que", "quem", "são",
    "se", "seja", "sejam", "sejamos", "sem", "ser", "serei", "seremos",
    "seria", "seriam", "seríamos", "seu", "seus", "só", "somos", "sou", "sua",
    "suas", "também", "te", "tem", "tém", "temos", "tenha", "tenham", "tenhamos",
    "tenho", "terá", "terão", "terei", "teremos", "teria", "teriam",
    "teríamos", "teu", "teus", "teve", "tinha", "tinham", "tínhamos", "tive",
    "tivemos", "tiver", "tivera", "tiveram", "tivéramos", "tiverem",
    "tivermos", "tivesse", "tivessem", "tivéssemos", "tu", "tua", "tuas",
    "um", "uma", "você", "vocês", "vos"
}

# ============================================================
# Funções de Limpeza
# ============================================================

def clean_text_for_bert(text: str) -> str:
    """
    Limpeza leve para o BERTimbau.
    Remove apenas URLs, normaliza menções (@user) e espaços extras.
    Mantém pontuação, maiúsculas/minúsculas e acentos.
    """
    if not isinstance(text, str):
        return ""

    # Substitui URLs por um token genérico ou remove
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    
    # Normaliza menções (o dataset já usa @user, mas garantimos)
    text = re.sub(r"@\w+", "@user", text)
    
    # Remove hashtags mantendo a palavra (ex: #ForaBolsonaro -> ForaBolsonaro)
    text = re.sub(r"#(\w+)", r"\1", text)
    
    # Remove espaços múltiplos
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def clean_text_for_tfidf(text: str) -> str:
    """
    Limpeza agressiva para Regressão Logística (TF-IDF).
    Converte para minúsculas, remove acentos, pontuação, números e stopwords.
    """
    if not isinstance(text, str):
        return ""

    # 1. Minúsculas
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

    # 3. Remove menções (@user) e hashtags (#)
    text = re.sub(r"@\w+|#\w+", "", text)

    # 4. Remove pontuação e caracteres especiais (mantém apenas letras e espaços)
    text = re.sub(r"[^\w\s]", "", text)

    # 5. Remove números
    text = re.sub(r"\d+", "", text)

    # 6. Remove acentos
    text = unidecode.unidecode(text)

    # 7. Tokenização simples e remoção de stopwords
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS_PT and len(t) > 1]

    return " ".join(tokens)


# ============================================================
# Pipeline Principal
# ============================================================

def process_dataset(dataset_path: Path, output_path: Path) -> Dict[str, Any]:
    """
    Carrega o dataset do disco, aplica as duas funções de limpeza
    e salva o novo dataset processado.
    """
    logger = setup_logging(name="preprocessing")
    from datasets import load_from_disk, DatasetDict, Dataset

    logger.info("Carregando dataset de: %s", dataset_path)
    ds = load_from_disk(str(dataset_path))

    def process_split(example):
        # Aplica a limpeza agressiva para TF-IDF
        example["text_tfidf"] = clean_text_for_tfidf(example["text"])
        
        # Aplica a limpeza leve para BERT
        example["text_bert"] = clean_text_for_bert(example["text"])
        
        return example

    logger.info("Aplicando pré-processamento nas partições...")
    ds_clean = DatasetDict()
    for split in ds.keys():
        logger.info("  Processando split: %s", split)
        # num_proc=2 para paralelizar (pode aumentar se o Colab aguentar)
        ds_clean[split] = ds[split].map(process_split, num_proc=2)

    logger.info("Salvando dataset processado em: %s", output_path)
    ensure_dir(output_path)
    ds_clean.save_to_disk(str(output_path))

    # Gera estatísticas para relatório
    stats = {}
    for split in ds_clean.keys():
        stats[split] = {
            "total_examples": len(ds_clean[split]),
            "empty_tfidf": sum(1 for t in ds_clean[split]["text_tfidf"] if not t.strip()),
            "empty_bert": sum(1 for t in ds_clean[split]["text_bert"] if not t.strip()),
        }
    
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Pré-processamento do ToLD-BR.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Caminho para o dataset carregado (ex: data/processed/told-br-binary).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/told-br-binary-clean",
        help="Caminho de saída para o dataset processado.",
    )
    args = parser.parse_args()

    logger = setup_logging(log_file="results/logs/preprocessing.log", name="preprocessing")
    
    stats = process_dataset(Path(args.input), Path(args.output))

    logger.info("=" * 60)
    logger.info("Resumo do Pré-processamento:")
    for split, info in stats.items():
        logger.info("  [%s] Total: %d | Vazios (TF-IDF): %d | Vazios (BERT): %d",
                    split, info["total_examples"], info["empty_tfidf"], info["empty_bert"])
    logger.info("=" * 60)

    # Salva estatísticas em JSON
    ensure_dir("results/metrics")
    save_results({"preprocessing_stats": stats}, "results/metrics", filename="preprocessing_stats.json")
    logger.info("Concluído com sucesso.")


if __name__ == "__main__":
    main()