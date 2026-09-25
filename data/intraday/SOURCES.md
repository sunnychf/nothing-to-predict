# Intraday source for the third anchor (transaction prices with a real bid-ask bounce)

Downloaded 2026-09-18 from Nasdaq's public sample directory (`download_intraday.sh` fetches the same file; `CHECKSUMS.md5` is its md5). The file is **not redistributed with the code**: it is Nasdaq's TotalView-ITCH sample data, published for developers at https://emi.nasdaq.com/ITCH/, so the released repository ships the download script, the extractor and this file, and each user fetches the day from Nasdaq. Nothing derived from it (the extracted executions `itch_trades_2019-12-30.npz`, the windows `results/intraday_windows.npz`) is shipped either; `results/intraday_windows_fingerprint.json` and `code/intraday_data.py --expect` confirm that a rebuild reproduces the paper's windows.

| File | Source | Content | Coverage |
|---|---|---|---|
| `12302019.NASDAQ_ITCH50.gz` (3.52 GB) | Nasdaq TotalView-ITCH 5.0 sample, https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/12302019.NASDAQ_ITCH50.gz | every order-book message of the Nasdaq market for one day: adds, cancels, executions, trades, for every listed and traded symbol | 2019-12-30, full day |

## What is used (`code/itch_trades.py`, `code/intraday_data.py`)

- Executions only: order-executed messages (E, and C when printable) priced at the resting order's price, and non-displayable trades (P); cross trades (Q, the opening and closing auctions) are not transactions in the sense of the bid-ask bounce and are dropped. Regular session 9:30–16:00 only. The resting order's side gives the trade direction: a resting bid hit is a sale at the bid, a resting ask lifted a purchase at the ask.
- One transaction per timestamp: fills sharing a nanosecond timestamp are one marketable order matched against several resting orders and are collapsed to the last fill's price.
- Symbols: after collapsing, the 40 symbols with the most transactions in the session whose day-low is at least $1 (sub-dollar stocks trade on a sub-penny tick); no other selection. Windows: consecutive non-overlapping blocks of 512 + 16 transactions in event time; 25 blocks per symbol, evenly spaced over the session's blocks (fewer if the symbol has fewer). Everything else is as for the daily anchor: K = 4 multiplicative sign-randomised copies per window (`numpy.random.default_rng(9000 + window_index + 100000 * draw)`), per-window scale `rsigma_*` = standard deviation of the context's simple returns.
- The trade directions are saved with the windows for the side-aware benchmark of `code/intraday_summary.py` (the mean of the last ask-side and the last bid-side transaction prices, an estimate of the quote midpoint); the models see values only.

## Pretraining overlap

The day is 2019-12-30; the models' corpora contain daily and lower-frequency financial series where they contain any, and no model's documentation lists Nasdaq order-book or trade-level data. The sign-randomised copies are new sequences by construction.
