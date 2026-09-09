checkpoint="./results/pretrained_finetune_mean_top_20_train_last_3_encoder/42"
output_dir="./results/pretrained_finetune_mean_top_20_train_last_3_encoder/42"
seed=42
device="cuda:0"
total_peptide_count=50000
top_k=100

python inference.py \
  --checkpoint "$checkpoint" \
  --output_dir "$output_dir" \
  --seed "$seed" \
  --device "$device" \
  --total-peptide-count "$total_peptide_count" \
  --top-k "$top_k"