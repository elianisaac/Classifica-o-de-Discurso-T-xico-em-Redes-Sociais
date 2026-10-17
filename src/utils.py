"""
utils.py
========

Funções auxiliares compartilhadas pelos scripts de treino, avaliação e
pré-processamento do projeto de detecção de discurso tóxico (ToLD-BR).

Cobre:
- Fixação de seeds e determinismo
- Detecção de dispositivo (CPU / CUDA)
- Configuração de logging
- I/O de configuração (YAML) e resultados (JSON)
- Métricas de avaliação padronizadas
- Utilidades de caminhos
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


# ============================================================
# Constantes do projeto
# ============================================================

SEED: int = 42
"""Seed global usada em todos os experimentos."""

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
"""Raiz do repositório (um nível acima de src/)."""

DEFAULT_LOG_FORMAT: str = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
DEFAULT_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"


# ============================================================
# Seeds e determinismo
# ============================================================

def set_seed(seed: int = SEED, deterministic: bool = True) -> None:
    """
    Fixa as seeds de todas as bibliotecas relevantes.

    Args:
        seed: valor da seed (padrão: SEED = 42).
        deterministic: se True, ativa modo determinístico do cuDNN
            (pequeno custo de desempenho, mas melhora reprodutibilidade).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # Variável de ambiente para operações determinísticas (PyTorch >= 1.8)
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                # Algumas versões/operações não suportam; seguimos com warn_only
                pass
    except ImportError:
        # torch não instalado (ex.: rodando só pré-processamento) — ok
        pass

    logging.getLogger(__name__).debug("Seed fixada em %d (deterministic=%s)", seed, deterministic)


# ============================================================
# Dispositivo (CPU / CUDA)
# ============================================================

def get_device(verbose: bool = True):
    """
    Retorna o dispositivo disponível (CUDA se houver, senão CPU).

    Args:
        verbose: se True, imprime o dispositivo e o nome da GPU (se houver).

    Returns:
        torch.device
    """
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        logger = logging.getLogger(__name__)
        if device.type == "cuda":
            logger.info("Dispositivo: CUDA (%s)", torch.cuda.get_device_name(0))
            logger.info("Versão do PyTorch: %s | CUDA: %s", torch.__version__, torch.version.cuda)
        else:
            logger.info("Dispositivo: CPU (nenhuma GPU CUDA detectada)")
            logger.info("Versão do PyTorch: %s", torch.__version__)

    return device


def describe_environment() -> Dict[str, Any]:
    """
    Coleta informações do ambiente para registro de reprodutibilidade.

    Returns:
        Dicionário com Python, PyTorch, CUDA, GPU etc.
    """
    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": sys.platform,
    }

    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda if torch.cuda.is_available() else None
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
    except ImportError:
        info["torch"] = None
        info["cuda_available"] = False

    try:
        import transformers

        info["transformers"] = transformers.__version__
    except ImportError:
        info["transformers"] = None

    try:
        import datasets

        info["datasets"] = datasets.__version__
    except ImportError:
        info["datasets"] = None

    try:
        import sklearn

        info["scikit_learn"] = sklearn.__version__
    except ImportError:
        info["scikit_learn"] = None

    return info


# ============================================================
# Logging
# ============================================================

def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path | str] = None,
    name: Optional[str] = None,
) -> logging.Logger:
    """
    Configura logging para o console e, opcionalmente, para arquivo.

    Args:
        level: nível de logging (padrão: INFO).
        log_file: caminho do arquivo de log (opcional). Se fornecido,
            cria o diretório pai se necessário.
        name: nome do logger. Se None, retorna o logger raiz.

    Returns:
        Logger configurado.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Evita duplicar handlers se chamado mais de uma vez
    if logger.handlers:
        return logger

    formatter = logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)

    # Handler de console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Handler de arquivo (opcional)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ============================================================
# I/O de configuração e resultados
# ============================================================

def load_config(path: Path | str) -> Dict[str, Any]:
    """
    Carrega um arquivo YAML de configuração.

    Args:
        path: caminho para o arquivo .yaml/.yml.

    Returns:
        Dicionário com a configuração.
    """
    import yaml

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_results(
    results: Dict[str, Any],
    output_dir: Path | str,
    filename: Optional[str] = None,
) -> Path:
    """
    Salva resultados em JSON, com timestamp no nome por padrão.

    Args:
        results: dicionário com métricas/configurações.
        output_dir: diretório de saída (criado se não existir).
        filename: nome do arquivo. Se None, usa run_<timestamp>.json.

    Returns:
        Caminho do arquivo salvo.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"run_{timestamp}.json"

    output_path = output_dir / filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logging.getLogger(__name__).info("Resultados salvos em %s", output_path)
    return output_path


def load_results(path: Path | str) -> Dict[str, Any]:
    """Carrega um arquivo JSON de resultados."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path | str) -> Path:
    """Garante que um diretório exista e retorna o Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ============================================================
# Métricas de avaliação
# ============================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    average: str = "binary",
) -> Dict[str, float]:
    """
    Calcula métricas padronizadas para a tarefa de classificação.

    Args:
        y_true: rótulos verdadeiros.
        y_pred: rótulos preditos.
        y_proba: probabilidades da classe positiva (opcional; necessário para AUC).
        average: 'binary' (classe tóxica = 1) ou 'macro'/'weighted' para multilabel.

    Returns:
        Dicionário com F1, precisão, recall e AUC (se y_proba fornecido).
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    metrics = {
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }

    if y_proba is not None:
        try:
            # Para binário, usa a probabilidade da classe positiva
            if average == "binary" and y_proba.ndim == 2:
                y_proba_pos = y_proba[:, 1]
            else:
                y_proba_pos = y_proba
            metrics["auc"] = float(roc_auc_score(y_true, y_proba_pos))
        except ValueError:
            metrics["auc"] = None

    return metrics


def format_metrics(metrics: Dict[str, float], title: Optional[str] = None) -> str:
    """
    Formata um dicionário de métricas como string legível.

    Args:
        metrics: dicionário retornado por compute_metrics.
        title: título opcional para o cabeçalho.

    Returns:
        String formatada.
    """
    lines = []
    if title:
        lines.append(f"=== {title} ===")
    for key, value in metrics.items():
        if value is None:
            lines.append(f"{key:>12}: N/A")
        else:
            lines.append(f"{key:>12}: {value:.4f}")
    return "\n".join(lines)


# ============================================================
# Utilidades gerais
# ============================================================

def timestamp() -> str:
    """Retorna um timestamp no formato YYYYMMDD_HHMMSS."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def print_section(title: str, logger: Optional[logging.Logger] = None) -> None:
    """Imprime (ou loga) um separador de seção."""
    line = f"\n{'=' * 60}\n{title}\n{'=' * 60}"
    if logger is not None:
        logger.info(line)
    else:
        print(line)


if __name__ == "__main__":
    # Demonstração / teste rápido
    setup_logging()
    set_seed()
    device = get_device()

    print_section("Ambiente", logging.getLogger(__name__))
    env = describe_environment()
    for k, v in env.items():
        print(f"  {k}: {v}")

    print_section("Métricas de exemplo", logging.getLogger(__name__))
    y_true = np.array([0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1])
    y_proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4],
                        [0.7, 0.3], [0.1, 0.9], [0.4, 0.6]])
    metrics = compute_metrics(y_true, y_pred, y_proba, average="binary")
    print(format_metrics(metrics, title="Exemplo"))