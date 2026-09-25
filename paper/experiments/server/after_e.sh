#!/bin/bash
# wait for runner e (timemoe, sundial, fincast) to end, then redo time-moe
until grep -q END ~/tsfm_rev/logs/falling_runner_$HOME/tsfm_rev/envs/fincast_v1/bin/python.log 2>/dev/null || ! ps aux | grep -q "[r]un_falling.sh <workdir>/envs/fincast_v1/bin/python 16 timemoe"; do sleep 30; done
rm -f ~/tsfm_rev/results_falling/falling_timemoe*.npz
bash ~/tsfm_rev/run_falling.sh ~/tsfm_rev/envs/fincast_v1/bin/python 16 timemoe
