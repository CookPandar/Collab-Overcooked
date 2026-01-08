python scripts/summarize_split_metrics.py \
  --log-root assets/data/batch_results/azure-gpt-4o/json \
  --train-data data/sft/gpt4o_gpt5_2/train_level12_Chef.jsonl \
  --dev-data data/sft/gpt4o_gpt5_2/dev_level12_Chef.jsonl \
  --test-data data/sft/gpt4o_gpt5_2/test_level12_Chef.jsonl \
  --plot-per-order-csv assets/data/batch_results/azure-gpt-4o/per_order_metrics.csv


python scripts/summarize_split_metrics.py \
  --log-root assets/data/batch_results/qwen2.5-7B-instruct/json \
  --train-data data/sft/gpt4o_gpt5_2/train_level12_Chef.jsonl \
  --dev-data data/sft/gpt4o_gpt5_2/dev_level12_Chef.jsonl \
  --test-data data/sft/gpt4o_gpt5_2/test_level12_Chef.jsonl \
  --plot-per-order-csv assets/data/batch_results/qwen2.5-7B-instruct/per_order_metrics.csv

python scripts/summarize_split_metrics.py \
  --log-root assets/data/batch_results/qwen2.5-sft-level1-sft-epoch-02/json \
  --train-data data/sft/gpt4o_gpt5_2/train_level12_Chef.jsonl \
  --dev-data data/sft/gpt4o_gpt5_2/dev_level12_Chef.jsonl \
  --test-data data/sft/gpt4o_gpt5_2/test_level12_Chef.jsonl \
  --plot-per-order-csv assets/data/batch_results/qwen2.5-sft-level1-sft-epoch-02/per_order_metrics.csv 

python scripts/summarize_split_metrics.py \
  --log-root assets/data/batch_results/qwen2.5-sft-level1-sft-gpt4o-gpt5_2-epoch-06/json \
  --train-data data/sft/gpt4o_gpt5_2/train_level12_Chef.jsonl \
  --dev-data data/sft/gpt4o_gpt5_2/dev_level12_Chef.jsonl \
  --test-data data/sft/gpt4o_gpt5_2/test_level12_Chef.jsonl \
  --plot-per-order-csv assets/data/batch_results/qwen2.5-sft-level1-sft-gpt4o-gpt5_2-epoch-06/per_order_metrics.csv

python scripts/summarize_split_metrics.py \
  --log-root assets/data/batch_results/qwen2.5-7B-sft-level12-epoch-06/json \
  --train-data data/sft/gpt4o_gpt5_2/train_level12_Chef.jsonl \
  --dev-data data/sft/gpt4o_gpt5_2/dev_level12_Chef.jsonl \
  --test-data data/sft/gpt4o_gpt5_2/test_level12_Chef.jsonl \
  --plot-per-order-csv assets/data/batch_results/qwen2.5-7B-sft-level12-epoch-06/per_order_metrics.csv

python scripts/summarize_split_metrics.py \
  --log-root assets/data/batch_results/azure-gpt-5_2/json \
  --train-data data/sft/gpt4o_gpt5_2/train_level12_Chef.jsonl \
  --dev-data data/sft/gpt4o_gpt5_2/dev_level12_Chef.jsonl \
  --test-data data/sft/gpt4o_gpt5_2/test_level12_Chef.jsonl \
  --plot-per-order-csv assets/data/batch_results/azure-gpt-5_2/per_order_metrics.csv

python scripts/summarize_split_metrics.py \
  --log-root assets/data/batch_results/qwen2.5-sft-level1-sft-gpt4o-gpt5_2-epoch-06/json \
  --plot-per-order-csv qwen2.5-sft-level1-sft-stage2=assets/data/batch_results/qwen2.5-sft-level1-sft-gpt4o-gpt5_2-epoch-06/per_order_metrics.csv \
   gpt-4o=assets/data/batch_results/azure-gpt-4o/per_order_metrics.csv qwen2.5-7B=assets/data/batch_results/qwen2.5-7B-instruct/per_order_metrics.csv \
   qwen2.5-7B-sft-stage1=assets/data/batch_results/qwen2.5-7B-sft-level12-epoch-06/per_order_metrics.csv  azure-gpt-5_2=assets/data/batch_results/azure-gpt-5_2/per_order_metrics.csv