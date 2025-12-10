# Ro-MCTS: Robust Monte Carlo Tree Search for RAG

Ro-MCTS is an extension of the [MCTS-RAG](README.md) framework designed to improve reasoning resilience against corrupted, irrelevant, or "poisoned" retrieval contexts. It integrates robustness mechanisms inspired by [RobustRAG](https://github.com/inspire-group/RobustRAG) directly into the MCTS search and evaluation loop.

## Key Features

### 1. Robust Leaf Evaluation (Isolate-then-Aggregate)
Instead of relying on a single generation's likelihood or a potentially hallucinated direct answer, Ro-MCTS evaluates answers using an **Isolate-then-Aggregate** strategy:
*   **Isolate**: The generated answer is verified against *each* retrieved document independently using a strict "Judge" prompt.
*   **Aggregate**: Individual verification scores are aggregated using robust statistics (e.g., Median) to filter out outliers caused by noisy documents.

### 2. Variance-Aware UCT
Standard MCTS selects nodes based on average reward. Ro-MCTS adds a **variance penalty** to the Upper Confidence Bound for Trees (UCT) formula. This discourages the search from following paths that yield highly inconsistent rewards (e.g., answers that only look correct under specific, rare contexts), favoring stable and consistently supported reasoning.

### 3. Robust Final Selection
When selecting the final answer from the search tree, Ro-MCTS groups solution nodes by their answer text and calculates the **Median** reward for each group. The answer with the highest median reward is selected, ensuring the chosen solution is robust to outlier evaluations.

## Usage

To enable robustness features, simply add the `--enable_robustness` flag to your run command.

### Basic Command

```bash
python run_src/MCTS_for_reasoning_with_rag.py \
    --dataset_name gsm8k \
    --test_json_filename test_mini \
    --num_rollouts 5 \
    --enable_robustness
```

### Configuration Arguments

You can fine-tune the robustness mechanisms with the following arguments:

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--enable_robustness` | `False` | Master flag to enable all Ro-MCTS features. |
| `--robust_sample_size` | `3` | Number of retrieved documents to evaluate against (lower = faster, higher = more robust). |
| `--robust_aggregation` | `mean` | Aggregation method for leaf evaluation scores. Options: `mean`, `median`, `trimmed_mean`. |
| `--uct_variance_weight` | `0.1` | Weight ($\lambda$) for the variance penalty in the UCT formula. Higher values make the search more risk-averse. |

### Example with Custom Settings

### Example with Custom Settings

```bash
python run_src/MCTS_for_reasoning_with_rag.py \
    --dataset_name gsm8k \
    --test_json_filename test \
    --num_rollouts 10 \
    --enable_robustness \
    --robust_sample_size 5 \
    --robust_aggregation trimmed_mean \
    --uct_variance_weight 0.2
```

### Full Run Command with `do_generate.py`

For a complete execution using the `do_generate.py` script (similar to the original repository's workflow), use the following command:

```bash
python run_src/do_generate.py \
    --dataset_name gsm8k \
    --test_json_filename test_mini \
    --api vllm \
    --model_ckpt meta-llama/Llama-3.1-8B-Instruct \
    --note robust_run \
    --half_precision \
    --num_rollouts 5 \
    --tensor_parallel_size 1 \
    --temperature 0.1 \
    --enable_robustness \
    --robust_sample_size 3 \
    --robust_aggregation mean \
    --uct_variance_weight 0.1 \
    --verbose
```

## Implementation Details

*   **`run_src/robust_eval.py`**: Contains the `RobustEvaluator` class which implements the verification and aggregation logic.
*   **`run_src/MCTS_backbone.py`**: Modified `MCTS_Searcher` to track reward variance (`Q_sq`) and compute the variance-aware UCT.
*   **`run_src/mcts_utils.py`**: Updated `find_best_solution` to perform robust final answer selection.
