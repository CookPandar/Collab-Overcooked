python scripts/summarize_split_metrics.py \
  --log-root assets/data/batch_results/azure-gpt-4o/json \
  --train-data data/sft/train_level12.jsonl \
  --dev-data data/sft/dev_level12.jsonl \
  --test-data data/sft/test_level12.jsonl \
  --per-order-csv assets/data/batch_results/azure-gpt-4o/per_order_metrics.csv \
  --split-csv assets/data/batch_results/azure-gpt-4o/split_metrics.csv 


python scripts/summarize_split_metrics.py \
  --log-root assets/data/batch_results/qwen2.5-7B-instruct/json \
  --train-data data/sft/train_level12.jsonl \
  --dev-data data/sft/dev_level12.jsonl \
  --test-data data/sft/test_level12.jsonl \
  --per-order-csv assets/data/batch_results/qwen2.5-7B-instruct/per_order_metrics.csv \
  --split-csv assets/data/batch_results/qwen2.5-7B-instruct/split_metrics.csv 

python scripts/summarize_split_metrics.py \
  --log-root assets/data/batch_results/qwen2.5-sft-level1-sft-epoch-02/json \
  --train-data data/sft/train_level12.jsonl \
  --dev-data data/sft/dev_level12.jsonl \
  --test-data data/sft/test_level12.jsonl \
  --per-order-csv assets/data/batch_results/qwen2.5-sft-level1-sft-epoch-02/per_order_metrics.csv \
  --split-csv assets/data/batch_results/qwen2.5-sft-level1-sft-epoch-02/split_metrics.csv 

python scripts/summarize_split_metrics.py \
  --log-root assets/data/batch_results/qwen2.5-sft-level1-sft-epoch-06/json \
  --train-data data/sft/train_level12.jsonl \
  --dev-data data/sft/dev_level12.jsonl \
  --test-data data/sft/test_level12.jsonl \
  --per-order-csv assets/data/batch_results/qwen2.5-sft-level1-sft-epoch-06/per_order_metrics.csv \
  --split-csv assets/data/batch_results/qwen2.5-sft-level1-sft-epoch-06/split_metrics.csv 


python scripts/summarize_split_metrics.py \
  --plot-per-order-csv qwen2.5-7B-sft-6epoch=assets/data/batch_results/qwen2.5-sft-level1-sft-epoch-06/per_order_metrics.csv \
   gpt-4o=assets/data/batch_results/azure-gpt-4o/per_order_metrics.csv qwen2.5-7B=assets/data/batch_results/qwen2.5-7B-instruct/per_order_metrics.csv  \
  --plot-output-dir assets/plots/per_order_comparison