#!/bin/bash
# Single-shot check-and-resubmit for the redundant Qwen3-14B DPO submissions
# (one job per slot, tracked independently). Meant to run every minute via
# cron, independent of any Claude Code session -- state persists in
# state.env between invocations.
# Policy (same as the original interactive watcher):
#  - on FAILED/CANCELLED/NODE_FAIL/TIMEOUT/OUT_OF_MEMORY before proving stable:
#    resubmit that slot, excluding any node the job has already crashed on.
#  - once a slot has been RUNNING continuously for 30 min, mark it stable and
#    cancel every other live slot as redundant.
set -u
WATCHDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_FILE="$WATCHDIR/state.env"
LOCK_FILE="$WATCHDIR/cron_watcher.lock"
SCRIPT="/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/qwen3/14B/DPO_AO/DPO-LR1e-7_Beta0.01-Qwen3-14B-SFT-LR3e-5-AO.sh"
BAD_NODES_FILE="/data/horse/ws/hama901h-BFTranslation/slurm_bad_nodes.txt"
STABLE_SECS=1800
SLOTS=(A B)

exec 9>"$LOCK_FILE"
flock -n 9 || { echo "$(date '+%F %T') another run still in progress, skipping"; exit 0; }

# --- state (defaults, overridden by state.env if present) ---
JOBID_A="" ; JOBID_B=""
ACCOUNT_A="p_neurasearch" ; ACCOUNT_B="p_scads_nas"
PREVSTATE_A="NONE" ; PREVSTATE_B="NONE"
RUN_START_A=0 ; RUN_START_B=0
STABLE_A=0 ; STABLE_B=0
[ -f "$STATE_FILE" ] && source "$STATE_FILE"

log() { echo "$(date '+%F %T') $*"; }

read_bad_nodes() {
  grep -vE '^\s*(#|$)' "$BAD_NODES_FILE" 2>/dev/null | tr -d ' \t' | paste -sd, -
}

resubmit() {
  local slot=$1
  local acct_var="ACCOUNT_$slot"
  local acct=${!acct_var}
  local exclude
  exclude=$(read_bad_nodes)
  local args=(--account="$acct")
  [ -n "$exclude" ] && args+=(--exclude="$exclude")
  local out newid
  out=$(sbatch "${args[@]}" "$SCRIPT" 2>&1)
  newid=$(echo "$out" | grep -oE 'Submitted batch job [0-9]+' | awk '{print $4}')
  if [ -z "$newid" ]; then
    log "slot $slot: RESUBMIT FAILED, sbatch output: $out"
    printf -v "JOBID_$slot" ""
    printf -v "PREVSTATE_$slot" "NONE"
    return
  fi
  log "slot $slot: resubmitted as job $newid (account $acct, exclude=${exclude:-none})"
  printf -v "JOBID_$slot" "%s" "$newid"
  printf -v "PREVSTATE_$slot" "PENDING"
  printf -v "RUN_START_$slot" "0"
}

add_exclude() {
  local nodelist=$1
  [ -z "$nodelist" ] && return
  case "$nodelist" in
    [Nn]one*|"") return ;;
  esac
  local expanded
  expanded=$(scontrol show hostnames "$nodelist" 2>/dev/null)
  local cur
  cur=$(read_bad_nodes)
  for n in $expanded; do
    [[ "$n" =~ ^c[0-9]+$ ]] || continue
    case ",$cur," in
      *,"$n",*) : ;;
      *)
        echo "$n" >> "$BAD_NODES_FILE"
        cur="$cur,$n"
        log "excluding node $n from future submissions, appended to $BAD_NODES_FILE"
        ;;
    esac
  done
}

any_other_stable() {
  local this=$1 s stable_var
  for s in "${SLOTS[@]}"; do
    [ "$s" = "$this" ] && continue
    stable_var="STABLE_$s"
    [ "${!stable_var}" = "1" ] && return 0
  done
  return 1
}

cancel_other_slots() {
  local this=$1 s jobid_var stable_var oj ost
  for s in "${SLOTS[@]}"; do
    [ "$s" = "$this" ] && continue
    jobid_var="JOBID_$s"
    oj=${!jobid_var}
    stable_var="STABLE_$s"
    [ "${!stable_var}" = "cancelled" ] && continue
    if [ -n "$oj" ]; then
      ost=$(squeue -j "$oj" -h -o "%T" 2>/dev/null)
      if [ -n "$ost" ]; then
        scancel "$oj"
        log "slot $s job $oj: CANCELLED (redundant, slot $this is stable)"
      fi
    fi
    printf -v "STABLE_$s" "cancelled"
  done
}

for slot in "${SLOTS[@]}"; do
  stable_var="STABLE_$slot"
  [ "${!stable_var}" = "cancelled" ] && continue

  jobid_var="JOBID_$slot"
  j=${!jobid_var}
  if [ -z "$j" ]; then
    log "slot $slot: no live job tracked, submitting"
    resubmit "$slot"
    continue
  fi

  st=$(squeue -j "$j" -h -o "%T" 2>/dev/null)
  if [ -z "$st" ]; then
    st=$(sacct -j "$j" -X -n -o State 2>/dev/null | awk '{print $1}')
    [ -z "$st" ] && st="UNKNOWN"
  fi

  prevstate_var="PREVSTATE_$slot"
  prev=${!prevstate_var}
  run_start_var="RUN_START_$slot"

  if [ "$st" != "$prev" ]; then
    log "slot $slot job $j: $prev -> $st"
    if [ "$st" = "RUNNING" ] && [ "${!run_start_var}" = "0" ]; then
      printf -v "RUN_START_$slot" "%s" "$(date +%s)"
    fi
    if [[ "$st" =~ ^(FAILED|CANCELLED|CANCELLED\+|NODE_FAIL|TIMEOUT|OUT_OF_MEMORY)$ ]]; then
      nodelist=$(sacct -j "$j" -X -n -o NodeList 2>/dev/null | tr -d ' ')
      add_exclude "$nodelist"
      if any_other_stable "$slot"; then
        log "slot $slot: not resubmitting, another slot already proven stable"
        printf -v "STABLE_$slot" "cancelled"
      else
        resubmit "$slot"
      fi
    else
      printf -v "PREVSTATE_$slot" "%s" "$st"
    fi
  fi

  # re-read in case resubmit() or the transition block changed things
  st=$(squeue -j "${!jobid_var}" -h -o "%T" 2>/dev/null)
  run_start_val_var="RUN_START_$slot"
  stable_val_var="STABLE_$slot"
  if [ "$st" = "RUNNING" ] && [ "${!stable_val_var}" = "0" ] && [ "${!run_start_val_var}" != "0" ]; then
    now=$(date +%s)
    elapsed=$(( now - ${!run_start_val_var} ))
    if [ "$elapsed" -ge "$STABLE_SECS" ]; then
      printf -v "STABLE_$slot" "1"
      log "slot $slot job ${!jobid_var}: STABLE after 30 min running"
      cancel_other_slots "$slot"
    fi
  fi
done

{
  echo "JOBID_A=$JOBID_A"
  echo "JOBID_B=$JOBID_B"
  echo "ACCOUNT_A=$ACCOUNT_A"
  echo "ACCOUNT_B=$ACCOUNT_B"
  echo "PREVSTATE_A=$PREVSTATE_A"
  echo "PREVSTATE_B=$PREVSTATE_B"
  echo "RUN_START_A=$RUN_START_A"
  echo "RUN_START_B=$RUN_START_B"
  echo "STABLE_A=$STABLE_A"
  echo "STABLE_B=$STABLE_B"
} > "$STATE_FILE"
