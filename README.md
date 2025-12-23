# KITE: Knowledge-augmented, Incremental Training for Enhanced SFT

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)

**KITE** is a general framework for constructing high-performance domain expert LLMs. It addresses the challenges of data scarcity for complex reasoning and catastrophic forgetting in domain adaptation through a novel Knowledge Graph (KG)-driven data synthesis pipeline and a "local-to-global" progressive training strategy.

## 🚀 Key Features

- **KG-Driven Data Synthesis**: Generates cross-document, multi-hop QA pairs using a domain knowledge graph.
- **Rare-Node Guided Exploration**: Prioritizes "long-tail" knowledge via weighted random walks to ensure diverse coverage.
- **Ambiguous Node Construction**: Masks entity names with attributes to force multi-step deduction and prevent shortcut learning.
- **Two-Stage Progressive Training**:
  - **Stage 1**: Single-document knowledge internalization.
  - **Stage 2**: Cross-document knowledge integration and complex reasoning.
- **Catastrophic Forgetting Mitigation**: Uses base model self-refinement and Parameter-Efficient Fine-Tuning (LoRA).

## 🏗️ Framework Architecture

The KITE framework operates in four main phases:

1.  **Knowledge Graph Construction**: detailed extraction of entities, relations, and attributes from domain corpora with provenance tracking.
2.  **Single-Document Data Synthesis**: Generating factual QA pairs from individual document chunks.
3.  **Cross-Document Data Synthesis**: Constructing complex reasoning paths across multiple documents using rare-node guided random walks and diversity constraints.
4.  **Progressive Training**: A curriculum learning approach that fine-tunes the model on single-document data first, followed by cross-document reasoning data.

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/KITE.git
cd KITE
pip install -r requirements.txt
```

*Note: Requires PyTorch, Transformers, PEFT, and NetworkX.*

## 🏃 Usage

### 1. Data Preparation
Prepare your domain corpus in JSON format:
```json
[
  {"id": "doc1", "content": "Text content..."},
  {"id": "doc2", "content": "Text content..."}
]
```

### 2. Knowledge Graph Construction
Extract entities and build the graph:
```bash
python scripts/build_kg.py \
    --input_data data/corpus.json \
    --output_dir data/kg \
    --model_name_or_path meta-llama/Llama-2-7b-hf
```

### 3. Data Synthesis
Generate SFT training data (Single & Cross-document):
```bash
python scripts/synthesize_data.py \
    --kg_path data/kg/graph.pkl \
    --output_dir data/sft_data \
    --mode all
```

### 4. Progressive Training
Run the two-stage training pipeline:

**Stage 1: Knowledge Internalization**
```bash
python scripts/train.py \
    --stage 1 \
    --data_path data/sft_data/single_doc.json \
    --output_dir checkpoints/stage1 \
    --use_lora True
```

**Stage 2: Knowledge Integration**
```bash
python scripts/train.py \
    --stage 2 \
    --model_path checkpoints/stage1 \
    --data_path data/sft_data/cross_doc.json \
    --output_dir checkpoints/final \
    --use_lora True
```

## 📊 Results

KITE achieves state-of-the-art results on domain-specific benchmarks in Linguistics and Law, significantly outperforming larger general-purpose models (e.g., GPT-3.5) while preserving general capabilities.

## 📜 Citation

If you use KITE in your research, please cite our paper:

```bibtex

```

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
