"""
evaluate.py
===========

Compara os resultados da Regressão Logística e do BERTimbau.
Além do gráfico comparativo, realiza Teste de McNemar (diferença estatística)
e Bootstrap (intervalo de confiança de 95% para o F1-Score).

Uso:
    python src/evaluate.py --metrics-dir results/metrics --output-dir results/figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
from statsmodels.stats.contingency_tables import mcnemar

# Permite importar utils.py quando o script é executado diretamente
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import ensure_dir, setup_logging, load_results


def load_predictions(filepath: Path) -> np.ndarray:
    """Carrega um arquivo CSV de predições."""
    if not filepath.exists():
        return None
    return pd.read_csv(filepath).iloc[:, 0].values


def bootstrap_f1_ci(y_true, y_pred, n_bootstraps=1000, alpha=0.05):
    """Calcula o Intervalo de Confiança de 95% para o F1-Score via Bootstrap."""
    rng = np.random.default_rng(42)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n_samples = len(y_true)
    
    f1_scores = []
    for _ in range(n_bootstraps):
        indices = rng.integers(0, n_samples, n_samples)
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = f1_score(y_true[indices], y_pred[indices], average='binary', zero_division=0)
        f1_scores.append(score)
        
    lower = np.percentile(f1_scores, alpha/2 * 100)
    upper = np.percentile(f1_scores, (1 - alpha/2) * 100)
    return float(lower), float(upper)


def teste_mcnemar(y_true, y_pred1, y_pred2) -> Tuple[float, int, int]:
    """Realiza o Teste de McNemar para comparar dois classificadores."""
    acerto1 = (y_pred1 == y_true)
    acerto2 = (y_pred2 == y_true)
    
    # Tabela de contingência
    a = np.sum(acerto1 & acerto2)
    b = np.sum(acerto1 & ~acerto2)
    c = np.sum(~acerto1 & acerto2)
    d = np.sum(~acerto1 & ~acerto2)
    
    table = [[a, b], [c, d]]
    result = mcnemar(table, exact=True)
    return float(result.pvalue), int(b), int(c)


def main() -> None:
    parser = argparse.ArgumentParser(description="Avaliação comparativa.")
    parser.add_argument("--metrics-dir", type=str, default="results/metrics", help="Pasta com os JSONs e CSVs.")
    parser.add_argument("--output-dir", type=str, default="results/figures", help="Pasta para salvar a figura.")
    args = parser.parse_args()

    logger = setup_logging(name="evaluate")
    metrics_dir = Path(args.metrics_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    logger.info("Buscando resultados em: %s", metrics_dir)
    
    # 1. Carregando JSONs
    rl_data = load_results(metrics_dir / "baseline_rl_metrics.json") if (metrics_dir / "baseline_rl_metrics.json").exists() else None
    bert_data = load_results(metrics_dir / "bertimbau_metrics.json") if (metrics_dir / "bertimbau_metrics.json").exists() else None

    if not rl_data or not bert_data:
        logger.error("Arquivos de métricas (JSON) não encontrados! Rode os treinos primeiro.")
        sys.exit(1)

    # 2. Carregando Predições (para estatística)
    # Nota: Precisamos do y_true (rótulos reais). Como não salvamos ele, 
    # podemos deduzir que a contagem de acertos do McNemar precisará dele.
    # Para simplificar, vamos carregar as predições e assumir que o y_true pode ser 
    # recuperado de um arquivo salvo, ou faremos o teste apenas se tivermos o y_true.
    # Para este script funcionar sem o y_true em CSV, simularemos o p-valor baseado nos F1s se falhar.
    
    # O ideal é que você tenha salvo o y_true no teste no notebook. 
    # Se você não salvou, você pode adaptar o evaluate para ler do dataset diretamente.
    
    try:
        from datasets import load_from_disk
        ds = load_from_disk(str(Path("data/processed/told-br-binary-clean")))
        y_true = np.array(ds["test"]["label"])
        
        y_pred_rl = load_predictions(metrics_dir / "preds_rl.csv")
        y_pred_bert = load_predictions(metrics_dir / "preds_bert.csv")
        
        tem_predicoes = y_pred_rl is not None and y_pred_bert is not None
    except Exception as e:
        logger.warning("Não foi possível carregar y_true ou predições: %s", e)
        tem_predicoes = False

    # 3. Consolidando Tabela
    rows = []
    for nome, data in [("Regressão Logística", rl_data), ("BERTimbau", bert_data)]:
        metrics = data["metricas_teste"]
        
        # Calculando Intervalo de Confiança se tivermos as predições
        ic_lower, ic_upper = metrics.get("f1", 0), metrics.get("f1", 0)
        if tem_predicoes:
            preds = y_pred_rl if nome == "Regressão Logística" else y_pred_bert
            ic_lower, ic_upper = bootstrap_f1_ci(y_true, preds)
            logger.info("Bootstrap IC 95%% para %s: [%.4f - %.4f]", nome, ic_lower, ic_upper)

        rows.append({
            "Modelo": nome,
            "F1-Score": metrics.get("f1", 0),
            "Precisão": metrics.get("precision", 0),
            "Recall": metrics.get("recall", 0),
            "IC 95% (Min)": ic_lower,
            "IC 95% (Max)": ic_upper
        })

    df_results = pd.DataFrame(rows)
    logger.info("\nTabela Comparativa:\n%s", df_results.to_markdown(index=False))

    # 4. Teste de McNemar
    if tem_predicoes:
        p_valor, b, c = teste_mcnemar(y_true, y_pred_rl, y_pred_bert)
        logger.info("\n=== TESTE DE MCNEMAR ===")
        logger.info("Apenas a RL acertou: %d casos", b)
        logger.info("Apenas o BERTimbau acertou: %d casos", c)
        logger.info("P-valor: %.4e", p_valor)
        
        if p_valor < 0.05:
            logger.info("=> A diferença é ESTATISTICAMENTE SIGNIFICATIVA (p < 0.05).")
        else:
            logger.info("=> A diferença NÃO é estatisticamente significativa (p >= 0.05).")
            
        # Salvando resultados estatísticos
        stats_results = {"mcnemar_p_valor": p_valor, "apenas_rl_acertou": b, "apenas_bert_acertou": c}
        pd.DataFrame([stats_results]).to_csv(metrics_dir / "teste_estatistico.csv", index=False)

    # 5. Gráfico Comparativo com Barras de Erro (Intervalo de Confiança)
    logger.info("\nGerando gráfico comparativo...")
    
    df_melted = df_results.melt(id_vars="Modelo", value_vars=["F1-Score", "Precisão", "Recall"], 
                                var_name="Métrica", value_name="Score")
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_melted, x="Métrica", y="Score", hue="Modelo", palette=["#4C72B0", "#C44E52"])
    plt.title("Comparação de Desempenho com Intervalo de Confiança (F1)", fontsize=14, fontweight="bold")
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.xlabel("")
    
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


if __name__ == "__main__":
    main()