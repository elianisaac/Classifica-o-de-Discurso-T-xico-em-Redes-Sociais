"""
train_bertimbau.py
==================

Realiza o fine-tuning do BERTimbau para classificação binária de toxicidade.

Uso:
    python src/train_bertimbau.py \
        --data data/processed/told-br-binary-clean \
        --model neuralmind/bert-base-portuguese-cased \
        --epochs 3 \
        --batch-size 16 \
        --output-dir results/models/bertimbau
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# Permite importar utils.py quando o script é executado diretamente
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import (
    compute_metrics,
    ensure_dir,
    format_metrics,
    get_device,
    save_results,
    set_seed,
    setup_logging,
)


def compute_hf_metrics(eval_pred):
    """Função de métricas exigida pelo Trainer do Hugging Face."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    acc = accuracy_score(labels, predictions)
    
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tuning do BERTimbau.")
    parser.add_argument("--data", type=str, default="data/processed/told-br-binary-clean", help="Caminho do dataset.")
    parser.add_argument("--model", type=str, default="neuralmind/bert-base-portuguese-cased", help="Modelo base.")
    parser.add_argument("--epochs", type=int, default=3, help="Número de épocas.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--max-length", type=int, default=128, help="Comprimento máximo dos tokens.")
    parser.add_argument("--output-dir", type=str, default="results/models/bertimbau", help="Pasta de saída.")
    args = parser.parse_args()

    # 1. Setup inicial
    logger = setup_logging(log_file="results/logs/train_bertimbau.log", name="train_bertimbau")
    set_seed(42)
    device = get_device()

    # 2. Verificação do dataset
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error("Dataset não encontrado em: %s", data_path)
        logger.error("Você rodou o load_data.py e o preprocessing.py?")
        sys.exit(1)

    logger.info("Carregando dataset de: %s", data_path)
    dataset = load_from_disk(str(data_path))
    
    # Remove colunas que o BERT não precisa e renomeia o texto limpo
    colunas_para_remover = [col for col in ["text", "text_tfidf"] if col in dataset["train"].column_names]
    dataset = dataset.remove_columns(colunas_para_remover)
    dataset = dataset.rename_column("text_bert", "text")

    # 3. Tokenização
    logger.info("Carregando tokenizador: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=args.max_length
        )

    logger.info("Tokenizando dados...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.set_format("torch")

    # 4. Carregando o modelo
    logger.info("Carregando modelo...")
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    # 5. Configuração do Treinamento (Corrigida para versões novas do transformers)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",         # Atualizado de evaluation_strategy para eval_strategy
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none"               # Desativa wandb/tensorboard
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_hf_metrics,
    )

    # 7. Treinamento
    logger.info("Iniciando treinamento...")
    trainer.train()

    # 8. Avaliação final no teste
    logger.info("Avaliando no conjunto de teste...")
    predictions_output = trainer.predict(tokenized_datasets["test"])
    
    y_pred = np.argmax(predictions_output.predictions, axis=-1)
    y_true = predictions_output.label_ids
    y_proba = torch.nn.functional.softmax(torch.tensor(predictions_output.predictions), dim=-1).numpy()

    metrics = compute_metrics(y_true, y_pred, y_proba, average="binary")
    logger.info("\n%s", format_metrics(metrics, title="Resultados no Teste (BERTimbau)"))

    # 9. Salvando resultados
    resultados_bert = {
        "modelo": "BERTimbau",
        "parametros": {
            "epochs": args.epochs, 
            "batch_size": args.batch_size, 
            "max_length": args.max_length
        },
        "metricas_teste": metrics
    }
    
    ensure_dir("results/metrics")
    save_results(resultados_bert, "results/metrics", filename="bertimbau_metrics.json")
    logger.info("Concluído com sucesso.")


if __name__ == "__main__":
    main()