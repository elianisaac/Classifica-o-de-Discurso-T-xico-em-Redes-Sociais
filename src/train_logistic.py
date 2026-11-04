"""
train_logistic.py
=================

Treina a baseline de Regressão Logística com TF-IDF para detecção de
discurso tóxico no dataset ToLD-BR.

Uso:
    python src/train_logistic.py --data data/processed/told-br-binary-clean --output results/models/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

import joblib
import pandas as pd
from datasets import load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# Permite importar utils.py quando o script é executado diretamente
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import (
    compute_metrics,
    ensure_dir,
    format_metrics,
    save_results,
    set_seed,
    setup_logging,
)


def load_data(data_path: Path, text_column: str = "text_tfidf"):
    """Carrega o dataset do disco e extrai as listas de textos e rótulos."""
    logger = setup_logging(name="train_logistic")
    logger.info("Carregando dados de: %s", data_path)
    
    ds = load_from_disk(str(data_path))
    
    # Converte para pandas para facilitar a manipulação
    df_train = ds["train"].to_pandas()
    df_test = ds["test"].to_pandas()
    
    X_train = df_train[text_column].tolist()
    y_train = df_train["label"].tolist()
    
    X_test = df_test[text_column].tolist()
    y_test = df_test["label"].tolist()
    
    logger.info("Treino: %d amostras | Teste: %d amostras", len(X_train), len(X_test))
    return X_train, y_train, X_test, y_test


def train_baseline(
    X_train, y_train, X_test, y_test, output_dir: Path
) -> Dict[str, Any]:
    """Treina o modelo, faz o GridSearch e avalia no teste."""
    logger = setup_logging(name="train_logistic")
    
    # 1. Criando o Pipeline (TF-IDF + Regressão Logística)
    # O Pipeline garante que o TF-IDF seja fitado apenas no treino e transforme o teste
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ('clf', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    # 2. Definindo a grade de hiperparâmetros (note o prefixo 'clf__' para acessar o modelo no pipeline)
    param_grid = {
        'clf__C': [0.1, 1.0, 10.0],
        'clf__solver': ['liblinear', 'lbfgs'],
        'clf__class_weight': [None, 'balanced']
    }
    
    # 3. Configurando o GridSearch
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    logger.info("Iniciando GridSearch (isso pode levar alguns minutos)...")
    grid_search.fit(X_train, y_train)
    
    logger.info("Melhores parâmetros: %s", grid_search.best_params_)
    logger.info("Melhor F1 (validação cruzada): %.4f", grid_search.best_score_)
    
    # 4. Avaliação no conjunto de teste
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)
    
    metrics = compute_metrics(y_test, y_pred, y_proba, average="binary")
    
    logger.info("\n%s", format_metrics(metrics, title="Resultados no Teste"))
    
    # 5. Salvando o modelo treinado
    ensure_dir(output_dir)
    model_path = output_dir / "logistic_regression_best.joblib"
    joblib.dump(best_model, model_path)
    logger.info("Modelo salvo em: %s", model_path)
    
    return {
        "melhores_parametros": grid_search.best_params_,
        "melhor_f1_cv": float(grid_search.best_score_),
        "metricas_teste": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina baseline de Regressão Logística.")
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/told-br-binary-clean",
        help="Caminho para o dataset limpo.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/models",
        help="Diretório para salvar o modelo treinado.",
    )
    args = parser.parse_args()

    logger = setup_logging(log_file="results/logs/train_logistic.log", name="train_logistic")
    set_seed(42)

    # Verifica se a pasta de dados existe
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error("Dataset não encontrado em: %s", data_path)
        logger.error("Você rodou o load_data.py e o preprocessing.py?")
        sys.exit(1)

    # Executa o fluxo
    X_train, y_train, X_test, y_test = load_data(data_path)
    results = train_baseline(X_train, y_train, X_test, y_test, Path(args.output))

    # Salva os resultados em JSON
    ensure_dir("results/metrics")
    save_results(results, "results/metrics", filename="baseline_rl_metrics.json")
    logger.info("Concluído com sucesso.")


if __name__ == "__main__":
    main()