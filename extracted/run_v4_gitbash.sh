#!/usr/bin/env bash
set -u

script_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$script_dir"

py -3 -u preflight_v4.py || exit $?

printf '\nCheck CAN_ENABLED in depth_cam/calib/fsm_v4/config.py before starting.\n'
printf 'CAN ON starts SEARCH with an in-place right turn; CAN OFF is monitor-only.\n'
read -r -p 'Start FSM v4 now? [y/N]: ' answer
case "$answer" in
    y|Y) ;;
    *) exit 0 ;;
esac

cd "$script_dir/depth_cam"
exec py -3 -u main_rec_v4.py
