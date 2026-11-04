"""
evaluate.py
===========

Compara os resultados da Regressão Logística e do BERTimbau.
Lê os arquivos `baseline_rl_metrics.json` e `bertimbau_metrics.json` da 
pasta de métricas e gera uma tabela e um gráfico comparativo.

Uso:
    python src/evaluate.py --metrics-dir results/metrics --output-dir results/figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Permite importar utils.py quando o script é executado diretamente
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import ensure_dir, setup_logging, load_results


def load_metrics_file(filepath: Path) -> Dict[str, Any]:
    """Carrega o JSON de métricas, ignorando se não existir."""
    if not filepath.exists():
        return None
    return load_results(filepath)


def main() -> None:
    parser = argparse.ArgumentParser(description="Avaliação comparativa.")
    parser.add_argument("--metrics-dir", type=str, default="results/metrics", help="Pasta com os JSONs.")
    parser.add_argument("--output-dir", type=str, default="results/figures", help="Pasta para salvar a figura.")
    args = parser.parse_args()

    logger = setup_logging(name="evaluate")
    metrics_dir = Path(args.metrics_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    # 1. Carregando os dados
    logger.info("Buscando resultados em: %s", metrics_dir)
    rl_data = load_metrics_file(metrics_dir / "baseline_rl_metrics.json")
    bert_data = load_metrics_file(metrics_dir / "bertimbau_metrics.json")

    if not rl_data and not bert_data:
        logger.error("Nenhum arquivo de métricas encontrado!")
        sys.exit(1)

    # 2. Consolidando em um DataFrame
    rows = []
    if rl_data:
        metrics = rl_data["metricas_teste"]
        rows.append({
            "Modelo": "Regressão Logística",
            "F1-Score": metrics.get("f1", 0),
            "Precisão": metrics.get("precision", 0),
            "Recall": metrics.get("recall", 0),
            "Acurácia": metrics.get("accuracy", 0)
        })
    
    if bert_data:
        metrics = bert_data["metricas_teste"]
        rows.append({
            "Modelo": "BERTimbau",
            "F1-Score": metrics.get("f1", 0),
            "Precisão": metrics.get("precision", 0),
            "Recall": metrics.get("recall", 0),
            "Acurácia": metrics.get("accuracy", 0)
        })

    df_results = pd.DataFrame(rows)
    
    # Imprime a tabela no terminal
    logger.info("\nTabela Comparativa:\n%s", df_results.to_markdown(index=False))

    # 3. Gerando o gráfico comparativo
    if len(df_results) > 0:
        logger.info("Gerando gráfico comparativo...")
        
        # Transformando o DataFrame para o formato longo (ideal para Seaborn)
        df_melted = df_results.melt(id_vars="Modelo", var_name="Métrica", value_name="Score")
        
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=df_melted, x="Métrica", y="Score", hue="Modelo", palette=["#4C72B0", "#C44E52"])
        
        plt.title("Comparação de Desempenho: Regressão Logística vs BERTimbau", fontsize=14, fontweight="bold")
        plt.ylim(0, 1.0)
        plt.ylabel("Score")
        plt.xlabel("")
        plt.legend(title="Modelo", loc="upper right")
        
        # Adicionando os valores nas barras
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.annotate(f"{height:.3f}",
                            (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom',
                            fontsize=9, xytext=(0, 3),
                            textcoords='offset points')

        plt.tight_layout()
        fig_path = output_dir / "comparacao_modelos.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        logger.info("Gráfico salvo em: %s", fig_path)

        # Opcional: Salvar a tabela em CSV para o relatório
        csv_path = metrics_dir / "comparacao_final.csv"
        df_results.to_csv(csv_path, index=False)
        logger.info("Tabela CSV salva em: %s", csv_path)

if __name__ == "__main__":
    main()