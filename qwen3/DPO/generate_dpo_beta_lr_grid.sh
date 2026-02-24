#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/home/hk-project-p0024043/hgf_ivw0083/ws/hkfswork/hgf_ivw0083-Post-training/finetuning/qwen3/DPO"
TEMPLATE_CFG="$BASE_DIR/config_dpo-n_LR8e8_Beta5.yaml"
TEMPLATE_SH="$BASE_DIR/dpo-n_LR8e8._Beta5.sh"

# Edit these sweeps as needed (or pass via env)
LRS=(${LRS:-"8e-8 1e-7 2e-7"})
BETAS=(${BETAS:-"0.1 0.5 1 2 5"})

lr_to_token() {
  local x="$1"
  if [[ "$x" == *"e-"* ]]; then
    echo "${x/e-/e}"     # 8e-8 -> 8e8
  elif [[ "$x" == *"."* ]]; then
    echo "${x/./p}"      # 0.0001 -> 0p0001
  else
    echo "$x"
  fi
}

beta_to_token() {
  local x="$1"
  echo "${x/./p}"        # 0.5 -> 0p5
}

[[ -f "$TEMPLATE_CFG" && -f "$TEMPLATE_SH" ]] || { echo "Template missing"; exit 1; }

for beta in "${BETAS[@]}"; do
  btoken="$(beta_to_token "$beta")"
  for lr in "${LRS[@]}"; do
    ltoken="$(lr_to_token "$lr")"

    out_cfg="$BASE_DIR/config_dpo-n_LR${ltoken}_Beta${btoken}.yaml"
    out_sh="$BASE_DIR/dpo-n_LR${ltoken}._Beta${btoken}.sh"

    cp "$TEMPLATE_CFG" "$out_cfg"
    cp "$TEMPLATE_SH" "$out_sh"

    # config updates
    sed -i "s|^learning_rate: .*|learning_rate: ${lr}|g" "$out_cfg"
    sed -i "s|^beta: .*|beta: ${beta}|g" "$out_cfg"
    sed -i "s|config_dpo-n_LR8e8_Beta5.yaml|config_dpo-n_LR${ltoken}_Beta${btoken}.yaml|g" "$out_cfg"
    sed -i "s|Beta5_LR8e-8|Beta${btoken}_LR${lr}|g" "$out_cfg"

    # slurm script updates
    sed -i "s|--job-name=dpo_norm_LR8e8_Beta5|--job-name=dpo_norm_LR${ltoken}_Beta${btoken}|g" "$out_sh"
    sed -i "s|config_dpo-n_LR8e8_Beta5.yaml|config_dpo-n_LR${ltoken}_Beta${btoken}.yaml|g" "$out_sh"
    sed -i "s|dpo-n_LR8e8._Beta5.sh|dpo-n_LR${ltoken}._Beta${btoken}.sh|g" "$out_sh" || true

    chmod +x "$out_sh"
    echo "Generated: $(basename "$out_cfg") | $(basename "$out_sh")"
  done
done
