#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/data/cat/ws/hama901h-Posttraining/finetuning/qwen3/DPO"
TEMPLATE_CFG="$BASE_DIR/template_config_LR5e8_Beta5.yaml"
TEMPLATE_SH="$BASE_DIR/template_dpo-n_LR5e8_Beta5.sh"

DEFAULT_LRS_STR="1e-8 5e-8 1e-7 5e-7 1e-9 5e-9"
DEFAULT_BETAS_STR="1 5 10"

# If USE_ENV_ONLY=1, use env as-is; otherwise merge env + defaults (unique)
LRS_STR="${LRS:-}"
BETAS_STR="${BETAS:-}"

if [[ "${USE_ENV_ONLY:-0}" == "1" ]]; then
  LRS_STR="${LRS_STR:-$DEFAULT_LRS_STR}"
  BETAS_STR="${BETAS_STR:-$DEFAULT_BETAS_STR}"
else
  LRS_STR="${LRS_STR} ${DEFAULT_LRS_STR}"
  BETAS_STR="${BETAS_STR} ${DEFAULT_BETAS_STR}"
fi

# Build unique arrays preserving order
declare -A _seen_lr=()
declare -A _seen_beta=()
LRS=()
BETAS=()

for x in $LRS_STR; do
  [[ -n "${_seen_lr[$x]:-}" ]] && continue
  _seen_lr[$x]=1
  LRS+=("$x")
done

for x in $BETAS_STR; do
  [[ -n "${_seen_beta[$x]:-}" ]] && continue
  _seen_beta[$x]=1
  BETAS+=("$x")
done

echo "Resolved LRS: ${LRS[*]}"
echo "Resolved BETAS: ${BETAS[*]}"

lr_to_token() {
  local x="$1"
  if [[ "$x" == *"e-"* ]]; then
    echo "${x/e-/e}"     # 5e-8 -> 5e8
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

# Escape replacement text for sed
sed_escape_repl() {
  printf '%s' "$1" | sed 's/[&|]/\\&/g'
}

# Safe in-file replacement (no-op if pattern not found)
safe_replace() {
  local file="$1" from="$2" to="$3"
  local to_esc
  to_esc="$(sed_escape_repl "$to")"
  sed -i "s|$from|$to_esc|g" "$file" || true
}

[[ -f "$TEMPLATE_CFG" && -f "$TEMPLATE_SH" ]] || { echo "Template missing"; exit 1; }

for beta in "${BETAS[@]}"; do
  btoken="$(beta_to_token "$beta")"
  for lr in "${LRS[@]}"; do
    ltoken="$(lr_to_token "$lr")"

    out_cfg="$BASE_DIR/config_dpo-n_LR${ltoken}_Beta${btoken}.yaml"
    out_sh="$BASE_DIR/dpo-n_LR${ltoken}_Beta${btoken}.sh"

    echo "Processing beta=${beta}, lr=${lr}"

    # prevent whole script from exiting on a single combo failure
    set +e
    {
      cp "$TEMPLATE_CFG" "$out_cfg"
      cp "$TEMPLATE_SH" "$out_sh"

      # config updates
      safe_replace "$out_cfg" '^learning_rate: .*' "learning_rate: ${lr}"
      safe_replace "$out_cfg" '^beta: .*' "beta: ${beta}"
      safe_replace "$out_cfg" 'config_LR5e8_Beta5.yaml' "config_dpo-n_LR${ltoken}_Beta${btoken}.yaml"
      safe_replace "$out_cfg" 'config_dpo-n_LR5e8_Beta5.yaml' "config_dpo-n_LR${ltoken}_Beta${btoken}.yaml"
      safe_replace "$out_cfg" 'Beta5_LR5e-8' "Beta${btoken}_LR${lr}"
      safe_replace "$out_cfg" 'Beta5_LR5e8' "Beta${btoken}_LR${lr}"

      # slurm script updates
      safe_replace "$out_sh" '--job-name=dpo_norm_LR5e8_Beta5' "--job-name=dpo_norm_LR${ltoken}_Beta${btoken}"
      safe_replace "$out_sh" 'config_LR5e8_Beta5.yaml' "config_dpo-n_LR${ltoken}_Beta${btoken}.yaml"
      safe_replace "$out_sh" 'config_dpo-n_LR5e8_Beta5.yaml' "config_dpo-n_LR${ltoken}_Beta${btoken}.yaml"
      safe_replace "$out_sh" 'dpo-n_LR5e8_Beta5.sh' "dpo-n_LR${ltoken}_Beta${btoken}.sh"
      safe_replace "$out_sh" 'dpo-n_LR5e8._Beta5.sh' "dpo-n_LR${ltoken}_Beta${btoken}.sh"

      chmod +x "$out_sh"
    }
    rc=$?
    set -e

    if [[ $rc -ne 0 ]]; then
      echo "FAILED combo beta=${beta}, lr=${lr} (rc=$rc), continuing..."
      continue
    fi

    echo "Generated: $(basename "$out_cfg") | $(basename "$out_sh")"
  done
done
