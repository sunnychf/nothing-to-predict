#!/bin/bash
# one config: arch corpus seed ; skip if its JSON exists (resumable)
a=$1; c=$2; s=$3; O=$HOME/tsfm_rev/results_arch
[ -f $O/arch_${a}_${c}_s${s}.json ] && exit 0
cd $HOME/tsfm_rev/code && OMP_NUM_THREADS=14 MKL_NUM_THREADS=14 $HOME/tsfm_rev/env/bin/python arch_train.py --arch $a --corpus $c --seed $s --steps 12000 --d 256 --layers 6 --device cpu --out-dir $O > $HOME/tsfm_rev/logs/arch_${a}_${c}_s${s}.log 2>&1
