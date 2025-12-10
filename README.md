<div align="center">
  <h1 align="center">Ro-MCTS RAG</h1>
  <p align="center">
    Trust-Aware and Explainable Multi-Step Retrieval-Augmented Reasoning
  </p>
  <p align="center">
    <strong>Ro-MCTS</strong> is a robustness-enhanced extension of <a href="#legacy-mcts-rag-baseline">MCTS-RAG</a> that stabilizes reasoning against retrieval noise found in complex, knowledge-intensive questions.
  </p>
</div>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.12%2B-blue.svg">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green.svg">
</p>

## Introduction

Multi-step reasoning systems like standard MCTS-RAG are powerful but fragile—a single misleading document retrieved early in the tree can propagate errors and distort the final answer. **Ro-MCTS** addresses this by integrating robust statistical principles directly into the search process.

### Key Features
1.  **Robust Leaf Evaluation (Isolate-then-Aggregate)**: 
    *   Instead of blindly accepting a generated answer, Ro-MCTS verifies it against *each* retrieved document independently using a "Strict Judge" prompt.
    *   Scores are aggregated using robust statistics (default: **Mean**) to minimize the impact of "poisoned" or irrelevant contexts.
2.  **Variance-Aware UCT Exploration**: 
    *   The search algorithm penalizes branches with high reward variance ($\lambda \sigma$), avoiding "lucky" paths that are unstable or hallucinated.
3.  **Robust Final Selection**: 
    *   Final answers are chosen based on the most consistent group score across all rollouts, rather than just the single highest path reward.

---

## Quick Start (Ro-MCTS)

To run the robust system, simply add the `--enable_robustness` flag to your generation script.

### Using `do_generate.py` (Recommended)

```bash
python run_src/do_generate.py \
    --dataset_name gsm8k \
    --test_json_filename test_mini \
    --api vllm \
    --model_ckpt meta-llama/Llama-3.1-8B-Instruct \
    --note romcts_run \
    --half_precision \
    --num_rollouts 5 \
    --temperature 0.1 \
    --enable_robustness \
    --robust_aggregation mean \
    --uct_variance_weight 0.1
```

### Configuration Options

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--enable_robustness` | `False` | Master flag to enable all Ro-MCTS features. |
| `--robust_sample_size` | `3` | Number of retrieved documents to compare against ($k$). |
| `--robust_aggregation` | `mean` | Aggregation method (`mean` or `median`). |
| `--uct_variance_weight` | `0.1` | Weight ($\lambda$) for the variance penalty in UCT. |

---

## Prerequisites

- Python 3.10+
- CUDA 12
- PyTorch
- `transformers`
- `vllm`

## Legacy MCTS-RAG Baseline

This repository also supports the original MCTS-RAG baseline. Use it to reproduce comparison results or for clean retrieval settings.

```bash
bash scripts/run_gsm8k_generator.sh
```
Or run `do_generate.py` **without** the `--enable_robustness` flag.

## Results

Ro-MCTS significantly outperforms the baseline in environments with retrieval noise.

| Model | Dataset | Baseline | Ro-MCTS |
| :--- | :--- | :--- | :--- |
| Llama 3.1 8B | FMT | 73.8 | **76.0** |
| Llama 3.1 8B | GPQA | 71.3 | **74.0** |
| Qwen2.5 7B | FMT | 68.2 | **69.6** |
| Qwen2.5 7B | GPQA | 64.6 | **65.7** |

## Implementation Details

*   **`run_src/robust_eval.py`**: Implements the `RobustEvaluator` class and the Isolate-then-Aggregate logic.
*   **`run_src/MCTS_backbone.py`**: Modified `MCTS_Searcher` with the variance-aware UCT formula.
*   **`run_src/mcts_utils.py`**: Updated `find_best_solution` for robust group-based answer selection.

## Citation

If you use MCTS-RAG, please cite:

```bibtex
@misc{hu2025mctsragenhancingretrievalaugmentedgeneration,
      title={MCTS-RAG: Enhancing Retrieval-Augmented Generation with Monte Carlo Tree Search}, 
      author={Yunhai Hu and Yilun Zhao and Chen Zhao and Arman Cohan},
      year={2025},
      eprint={2503.20757},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.20757}, 
}
```
