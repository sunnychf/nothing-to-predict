# ETTh1, for the Time-MoE harness validation only

`ETTh1.csv` is the Electricity Transformer Temperature dataset of Zhou et al. (2021), from
https://github.com/zhouhaoyi/ETDataset (ETT-small), downloaded 2026-09-19 (`download_etth1.sh`;
sha256 in `CHECKSUMS.sha256`). It is used by `code/validate_timemoe.py` to reproduce the zero-shot
ETTh1 numbers that the Time-MoE paper reports for Time-MoE_large, with the decoding loop this paper
wrote for the model (Appendix, "The Time-MoE harness"), in the protocol of the Time-MoE repository's
evaluation code (train-rows standardisation, stride-one windows over the test segment, context
512/1024/2048/3072 for horizons 96/192/336/720). It is not used anywhere else in the paper and is not
redistributed with the code.
