# KITE: Knowledge-augmented, Incremental Training for Enhanced SFT

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

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

    ![Data Synthesis Pipeline](figure/data_syn.png)

4.  **Progressive Training**: A curriculum learning approach that fine-tunes the model on single-document data first, followed by cross-document reasoning data.

    ![Two-Stage Progressive Training](figure/two_stage.png)

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/KITE.git
```

## 🏃 Usage

### 1. Data Preparation
Prepare your domain corpus in JSONL format:
```jsonl
  {"id": "doc1", "content": "Text content..."},
  {"id": "doc2", "content": "Text content..."}
```

### 2. Knowledge Graph Construction
Use langextract to extract entities and build the graph:
```bash
python extract_graph.py
```
Convert JSON file to Neo4j:
```bash
python json2neo4j.py
```

### 3. Data Synthesis
Identify rare nodes:
```bash
python rare_node.py
```
Cross-document weighted random walk:
```bash
python cross_doc_walk.py
```
Generate questions:
```bash
python generate_qa.py
```
Generate answers:
```bash
python generate_answer.py
```



## 📜 Citation

If you use KITE in your research, please cite our paper:

```bibtex

```

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.




