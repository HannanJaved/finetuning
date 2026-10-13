#!/bin/bash
# Single tick of the integrated watch loop (called every ~90s by the outer Monitor loop).
# Does three things:
#  1. Processes the ongoing 38-node recheck sweep (+ c40): any real FAILED/COMPLETED
#     result gets logged once and, if failed, the node is added to the shared
#     bad-nodes file (idempotent).
#  2. If the bad-nodes file grew this tick, cancels any qwen3-14b job (tracked by
#     the cron watcher) whose SchedNodeList/NodeList overlaps a newly-bad node --
#     the cron watcher (runs every minute, reads the file fresh) will resubmit it
#     with the updated exclude list on its next tick.
#  3. For each qwen3-14b job's currently assigned/scheduled node(s): looks for
#     another user's job that's been RUNNING there for >2 minutes as corroborating
#     evidence of health. If none exists, submits a lightweight CPU-only (no GPU,
#     so it doesn't contend with qwen3's own GPU allocation on that node) 2-minute
#     verification probe to that specific node, once per node per watch lifetime.
set -u
WATCHDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCK_FILE="$WATCHDIR/integrated_watch.lock"
BAD_NODES_FILE="/data/horse/ws/hama901h-BFTranslation/slurm_bad_nodes.txt"
RECHECK_MAP="$WATCHDIR/recheck_all_excluded_map.tsv"
QWEN_STATE="$WATCHDIR/state.env"
LOGDIR="/data/horse/ws/hama901h-BFTranslation/node_health_recheck"
PROCESSED_FILE="$WATCHDIR/integrated_watch_processed.txt"
VERIFIED_NODES_FILE="$WATCHDIR/integrated_watch_verified_nodes.txt"
touch "$PROCESSED_FILE" "$VERIFIED_NODES_FILE"

exec 8>"$LOCK_FILE"
flock -n 8 || exit 0

read_bad_nodes() { grep -vE '^\s*(#|$)' "$BAD_NODES_FILE" 2>/dev/null | tr -d ' \t' | paste -sd, -; }

add_bad_node() {
  local n=$1
  local cur; cur=$(read_bad_nodes)
  case ",$cur," in
    *,"$n",*) return 1 ;;
    *) echo "$n" >> "$BAD_NODES_FILE"; echo "$(date '+%T') NEW BAD NODE DISCOVERED: $n"; return 0 ;;
  esac
}

already_processed() { grep -qxF "$1" "$PROCESSED_FILE"; }
mark_processed() { echo "$1" >> "$PROCESSED_FILE"; }
already_verified() { grep -qxF "$1" "$VERIFIED_NODES_FILE"; }
mark_verified() { echo "$1" >> "$VERIFIED_NODES_FILE"; }

before_count=$(read_bad_nodes | tr ',' '\n' | grep -c .)

# --- 1. process recheck sweep + c40 ---
{ cat "$RECHECK_MAP" 2>/dev/null; printf "c40\t4203319\n"; } | while IFS=$'\t' read -r n j; do
  [ -z "${j:-}" ] && continue
  already_processed "job:$j" && continue
  st=$(sacct -j "$j" -X -n -o State 2>/dev/null)
  [[ "$st" == *PENDING* || "$st" == *RUNNING* ]] && continue
  mark_processed "job:$j"
  if [[ "$st" == *COMPLETED* ]]; then
    echo "$(date '+%T') RECHECK: $n ($j) COMPLETED -- appears healthy again"
  else
    echo "$(date '+%T') RECHECK: $n ($j) $st"
    add_bad_node "$n" || true
  fi
done

after_count=$(read_bad_nodes | tr ',' '\n' | grep -c .)

# --- 2. if new bad nodes appeared, cancel any qwen3 job scheduled onto one ---
if [ "$after_count" -gt "$before_count" ]; then
  bad_now=$(read_bad_nodes)
  if [ -f "$QWEN_STATE" ]; then
    source "$QWEN_STATE"
    for slot in A B; do
      jobid_var="JOBID_$slot"
      j=${!jobid_var:-}
      [ -z "$j" ] && continue
      sched=$(scontrol show job "$j" 2>/dev/null | grep -oP '(SchedNodeList|NodeList)=\K\S+' | grep -v '(null)' | head -1)
      [ -z "$sched" ] && continue
      expanded=$(scontrol show hostnames "$sched" 2>/dev/null)
      for n in $expanded; do
        case ",$bad_now," in
          *,"$n",*)
            st=$(squeue -j "$j" -h -o "%T" 2>/dev/null)
            if [ -n "$st" ]; then
              scancel "$j"
              echo "$(date '+%T') PROACTIVE: cancelled qwen3 slot $slot job $j -- scheduled onto newly-bad node $n (cron will resubmit)"
            fi
            ;;
        esac
      done
    done
  fi
fi

# --- 3. corroborating-evidence check for qwen3's current node(s) ---
if [ -f "$QWEN_STATE" ]; then
  source "$QWEN_STATE"
  for slot in A B; do
    jobid_var="JOBID_$slot"
    j=${!jobid_var:-}
    [ -z "$j" ] && continue
    st=$(squeue -j "$j" -h -o "%T" 2>/dev/null)
    [ "$st" != "RUNNING" ] && continue
    nodelist=$(squeue -j "$j" -h -o "%N" 2>/dev/null)
    [ -z "$nodelist" ] && continue
    expanded=$(scontrol show hostnames "$nodelist" 2>/dev/null)
    for n in $expanded; do
      already_verified "$n" && continue
      mark_verified "$n"
      other=$(squeue -w "$n" -t RUNNING -h -o "%i %M" 2>/dev/null | awk -v me="$j" '$1!=me')
      corroborated=0
      while read -r oid elapsed; do
        [ -z "$oid" ] && continue
        secs=0
        if [[ "$elapsed" =~ ^([0-9]+):([0-9]+):([0-9]+)$ ]]; then
          secs=$(( 10#${BASH_REMATCH[1]}*3600 + 10#${BASH_REMATCH[2]}*60 + 10#${BASH_REMATCH[3]} ))
        elif [[ "$elapsed" =~ ^([0-9]+):([0-9]+)$ ]]; then
          secs=$(( 10#${BASH_REMATCH[1]}*60 + 10#${BASH_REMATCH[2]} ))
        fi
        [ "$secs" -ge 120 ] && corroborated=1
      done <<< "$other"
      if [ "$corroborated" = "1" ]; then
        echo "$(date '+%T') slot $slot node $n: corroborated healthy by another job already running >2min"
      else
        out=$(sbatch --account=p_scads_nas --partition=capella --nodelist="$n" -N1 \
          --gres=gpu:0 --cpus-per-task=1 --mem=1G --time=00:02:00 \
          --output="$LOGDIR/verify_${n}_%j.out" --error="$LOGDIR/verify_${n}_%j.err" \
          --wrap="hostname; ls /data/horse >/dev/null && echo FS_OK; echo NODE_HEALTH_OK" 2>&1)
        jid=$(echo "$out" | grep -oE 'Submitted batch job [0-9]+' | awk '{print $4}')
        echo "$(date '+%T') slot $slot node $n: no corroborating job found, submitted independent verify probe -> ${jid:-FAILED: $out}"
      fi
    done
  done
fi
