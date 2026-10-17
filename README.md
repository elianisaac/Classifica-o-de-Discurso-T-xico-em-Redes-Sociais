# Detecção de Discurso Tóxico em Português Brasileiro: BERTimbau vs. Regressão Logística

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Status](https://img.shields.io/badge/status-em%20desenvolvimento-yellowgreen)

## 📋 Sobre o projeto

Este repositório contém o código, os experimentos e os resultados de uma pesquisa acadêmica que **compara o desempenho do BERTimbau com um modelo de Regressão Logística na tarefa de detecção automática de discurso tóxico em português brasileiro**, utilizando o dataset [ToLD-BR](https://github.com/JAugusto97/ToLD-BR).

O objetivo é avaliar se o ganho de desempenho de um modelo neural pré-treinado justifica seu maior custo computacional e sua menor interpretabilidade em relação a uma baseline clássica, no contexto do português brasileiro.

## 🎯 Objetivos

- **Geral:** Comparar o desempenho do BERTimbau e da Regressão Logística na detecção de discurso tóxico no dataset ToLD-BR.
- **Específicos:**
  - Pré-processar o corpus de acordo com as exigências de cada abordagem.
  - Treinar uma baseline de Regressão Logística com representação TF-IDF.
  - Realizar fine-tuning do BERTimbau para a mesma tarefa.
  - Avaliar ambos os modelos com métricas padronizadas (F1, precisão, recall).
  - Analisar os erros de classificação de cada modelo.

## 📚 Dataset

O **ToLD-BR** (Toxic Language Dataset for Brazilian Portuguese) é um dataset anotado manualmente com tweets em português brasileiro, contendo rótulos para:

- Toxicidade geral (binário)
- Categorias específicas: racismo, homofobia, misoginia, xenofobia, etc.

- 🔗 Repositório oficial: [ToLD-BR](https://github.com/JAugusto97/ToLD-BR)
- 📄 Artigo associado: Leite et al. (2020) — *Toxic Language Detection in Social Media for Brazilian Portuguese*

> **Nota:** o dataset não está incluído neste repositório por questões de licença e tamanho. Siga as instruções do repositório oficial para obtê-lo e coloque os arquivos em `data/raw/`.

## 🗂️ Estrutura do repositório

├── data/
│ ├── processed/ # Dados convertidos para disco (Parquet/Arrow)
│ │ └── told-br-binary/ # Gerado por src/load_data.py (não versionado)
│ └── README.md # Nota sobre a origem dos dados (Hugging Face)
├── notebooks/
│ ├── 00_setup_colab.ipynb # Setup do ambiente no Google Colab (GPU)
│ ├── 01_exploracao.ipynb # Análise exploratória do dataset
│ ├── 02_baseline_rl.ipynb # Regressão Logística + TF-IDF
│ └── 03_bertimbau.ipynb # Fine-tuning do BERTimbau
├── src/
│ ├── load_data.py # Carregamento do ToLD-BR via Hugging Face
│ ├── preprocessing.py # Limpeza, normalização e tokenização
│ ├── train_logistic.py # Pipeline da Regressão Logística
│ ├── train_bertimbau.py # Pipeline do BERTimbau
│ ├── evaluate.py # Métricas e comparação
│ └── utils.py # Funções auxiliares (seeds, device, logging)
├── results/
│ ├── figures/ # Gráficos gerados
│ ├── metrics/ # Tabelas de métricas (.csv / .json)
│ └── models/ # Modelos treinados e checkpoints (não versionados)
├── docs/
│ └── paper.pdf # Versão final do trabalho (quando disponível)
├── requirements.txt
├── requirements-colab.txt # Dependências compatíveis com o runtime do Colab
├── .gitignore
└── README.md

## 🧪 Reprodutibilidade

Este projeto foi desenhado para que os experimentos possam ser reproduzidos de forma consistente

### Fonte dos dados

- O dataset **ToLD-BR** é versionado publicamente no Hugging Face em [`JAugusto97/told-br`](https://huggingface.co/datasets/JAugusto97/told-br), garantindo que todos os experimentos usem exatamente a mesma versão dos dados.
- O carregamento é centralizado em `src/load_data.py`, que:
  - usa `load_dataset("JAugusto97/told-br", variant, trust_remote_code=True)`;
  - recarrega do disco com `load_from_disk()` quando a pasta já existe.
- Versão recomendada da biblioteca `datasets`: `>=2.14` (com fallback para `<3.0` caso o *loading script* falhe em versões mais recentes).

### Seeds e determinismo

- Todas as seeds são fixadas em `SEED = 42` (definidas em `src/utils.py`):

```python
import random, numpy as np, torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

```