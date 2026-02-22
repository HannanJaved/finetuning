#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./generate_lr_grid.sh 5e-6 1e-5 2e-5 5e-5
# If none provided, defaults are used.
LRS=("$@")
if [ ${#LRS[@]} -eq 0 ]; then
  LRS=("1e-3" "5e-3" "1e-4" "5e-4" "1e-5")
fi

BASE_DIR="/home/hk-project-p0024043/hgf_ivw0083/ws/hkfswork/hgf_ivw0083-Post-training/finetuning/datamix-90-40-9B"
TEMPLATE_CONFIG="$BASE_DIR/config_olmo3_sft_Lr5e5.yaml"
TEMPLATE_SH="$BASE_DIR/sft_olmo3_Lr5e5.sh"

if [[ ! -f "$TEMPLATE_CONFIG" || ! -f "$TEMPLATE_SH" ]]; then
  echo "Template files not found in $BASE_DIR"
  exit 1
fi

lr_to_token() {
  # 5e-5 -> 5e5, 1e-6 -> 1e6, 0.0001 -> 0p0001
  local lr="$1"
  if [[ "$lr" == *"e-"* ]]; then
    echo "${lr/e-/e}"
  elif [[ "$lr" == *"."* ]]; then
    echo "${lr/./p}"
  else
    echo "$lr"
  fi
}

for lr in "${LRS[@]}"; do
  token="$(lr_to_token "$lr")"

  out_cfg="$BASE_DIR/config_olmo3_sft_Lr${token}.yaml"
  out_sh="$BASE_DIR/sft_olmo3_Lr${token}.sh"

  cp "$TEMPLATE_CONFIG" "$out_cfg"
  cp "$TEMPLATE_SH" "$out_sh"

  # Update YAML learning rate and naming/path fields
  sed -i "s|^learning_rate: .*|learning_rate: ${lr}|g" "$out_cfg"
  sed -i "s|config_olmo3_sft_Lr5e5.yaml|config_olmo3_sft_Lr${token}.yaml|g" "$out_cfg"
  sed -i "s|olmo3-sft/Lr5e5/|olmo3-sft/Lr${token}/|g" "$out_cfg"

  # Update SLURM script job name + config reference
  sed -i "s|--job-name=olmo3-Lr5e5-datamix90-40-9B|--job-name=olmo3-Lr${token}-datamix90-40-9B|g" "$out_sh"
  sed -i "s|config_olmo3_sft_Lr5e5.yaml|config_olmo3_sft_Lr${token}.yaml|g" "$out_sh"
  sed -i "s|sft_olmo3_Lr5e5.sh|sft_olmo3_Lr${token}.sh|g" "$out_sh" || true

  chmod +x "$out_sh"
  echo "Generated:"
  echo "  $out_cfg"
  echo "  $out_sh"
done
