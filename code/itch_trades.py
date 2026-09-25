"""Extract transaction prices for a list of symbols from a Nasdaq TotalView-ITCH 5.0 day file
(the public sample files at emi.nasdaq.com/ITCH/, gzip; format per Nasdaq's ITCH 5.0
specification). One streaming pass; nothing but the executions of the chosen symbols is kept.

    python itch_trades.py 12302019.NASDAQ_ITCH50.gz --symbols SIRI,AMD,... --out itch_trades_2019-12-30.npz

Messages used: R (stock directory: locate code -> symbol), A/F (add order: order ref -> price,
side, symbol), U (replace: new ref inherits the symbol, takes the new price), E (order executed:
the resting order's price is the trade price; the aggressor is on the opposite side, so the
trade is at the ask when a resting sell is hit), C (executed with a different price; kept only
when printable), P (non-displayable trade, carries its own price and the resting side). Cross
trades (Q) are not trades in the sense of the bid-ask bounce and are dropped. Prices are in
units of 1e-4 dollars; timestamps in nanoseconds since midnight. Regular session only
(9:30-16:00). Output: per symbol, arrays of (t_ns, price, resting_side) for every execution,
plus the day's stock directory count and the totals per message type.
"""
import argparse, gzip, struct, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("--symbols", required=True, help="comma-separated tickers, or ALL to keep every symbol (the whole book is tracked; about 2 GB of memory)")
ap.add_argument("--out", required=True)
ap.add_argument("--max-bytes", type=int, default=0, help="stop after this many compressed bytes (testing)")
ap.add_argument("--all-day", action="store_true", help="keep executions outside 9:30-16:00 too (testing)")
args = ap.parse_args()
want = None if args.symbols.upper() == "ALL" else set(s.strip().upper() for s in args.symbols.split(","))

T_OPEN, T_CLOSE = 9 * 3600 * 10**9 + 30 * 60 * 10**9, 16 * 3600 * 10**9
if args.all_day: T_OPEN, T_CLOSE = 0, 24 * 3600 * 10**9
locate2sym = {}
orders = {}                      # order ref -> (price, side, sym) for wanted symbols only
out = {}                         # sym -> list of (t, price, side)
counts = {}
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def ts(b):                       # 6-byte big-endian nanoseconds
    return int.from_bytes(b, "big")

t0 = time.time(); nmsg = 0
with gzip.open(args.path, "rb") as f:
    while True:
        try:
            hdr = f.read(2)
            if len(hdr) < 2: break
            (ln,) = struct.unpack(">H", hdr)
            m = f.read(ln)
        except EOFError:                                  # truncated stream (partial download)
            log("compressed stream ended early"); break
        if len(m) < ln: break
        nmsg += 1; typ = m[0:1]
        counts[typ] = counts.get(typ, 0) + 1
        if typ == b"R":                                   # stock directory
            locate = struct.unpack(">H", m[1:3])[0]; sym = m[11:19].decode().strip()
            locate2sym[locate] = sym
        elif typ in (b"A", b"F"):
            locate = struct.unpack(">H", m[1:3])[0]; sym = locate2sym.get(locate)
            if want is None or sym in want:
                ref = struct.unpack(">Q", m[11:19])[0]; side = m[19:20]; shares = struct.unpack(">I", m[20:24])[0]; price = struct.unpack(">I", m[32:36])[0]
                orders[ref] = [price, side, sym, shares]
        elif typ == b"U":                                 # replace: keeps side and symbol
            old = struct.unpack(">Q", m[11:19])[0]
            o = orders.pop(old, None)
            if o is not None:
                new = struct.unpack(">Q", m[19:27])[0]; shares = struct.unpack(">I", m[27:31])[0]; price = struct.unpack(">I", m[31:35])[0]
                orders[new] = [price, o[1], o[2], shares]
        elif typ == b"E":
            ref = struct.unpack(">Q", m[11:19])[0]; o = orders.get(ref)
            if o is not None:
                t = ts(m[5:11])
                if T_OPEN <= t <= T_CLOSE: out.setdefault(o[2], []).append((t, o[0], o[1]))
                o[3] -= struct.unpack(">I", m[19:23])[0]
                if o[3] <= 0: del orders[ref]
        elif typ == b"C":
            ref = struct.unpack(">Q", m[11:19])[0]; o = orders.get(ref)
            if o is not None:
                if m[31:32] == b"Y":
                    t = ts(m[5:11]); price = struct.unpack(">I", m[32:36])[0]
                    if T_OPEN <= t <= T_CLOSE: out.setdefault(o[2], []).append((t, price, o[1]))
                o[3] -= struct.unpack(">I", m[19:23])[0]
                if o[3] <= 0: del orders[ref]
        elif typ == b"X":                                 # partial cancel
            ref = struct.unpack(">Q", m[11:19])[0]; o = orders.get(ref)
            if o is not None:
                o[3] -= struct.unpack(">I", m[19:23])[0]
                if o[3] <= 0: del orders[ref]
        elif typ == b"D":                                 # delete
            orders.pop(struct.unpack(">Q", m[11:19])[0], None)
        elif typ == b"P":
            locate = struct.unpack(">H", m[1:3])[0]; sym = locate2sym.get(locate)
            if want is None or sym in want:
                t = ts(m[5:11]); side = m[19:20]; price = struct.unpack(">I", m[32:36])[0]
                if T_OPEN <= t <= T_CLOSE: out.setdefault(sym, []).append((t, price, side))
        if nmsg % 20_000_000 == 0:
            log(f"{nmsg / 1e6:.0f}M messages, {len(orders)} live orders kept, {sum(len(v) for v in out.values())} executions, {time.time() - t0:.0f}s")
        if args.max_bytes and f.fileobj.tell() >= args.max_bytes: break
log(f"done: {nmsg / 1e6:.1f}M messages in {time.time() - t0:.0f}s; directory {len(locate2sym)} symbols; "
    f"types {dict(sorted(((k.decode(), v) for k, v in counts.items()), key=lambda kv: -kv[1])[:12])}")
save = {}
for sym, rows in out.items():
    rows.sort()
    save[f"{sym}/t"] = np.array([r[0] for r in rows], dtype=np.int64)
    save[f"{sym}/p"] = np.array([r[1] for r in rows], dtype=np.int64)
    save[f"{sym}/side"] = np.array([1 if r[2] == b"B" else -1 for r in rows], dtype=np.int8)
    log(f"  {sym}: {len(rows)} executions, price {save[f'{sym}/p'].min() / 1e4:.2f}-{save[f'{sym}/p'].max() / 1e4:.2f}")
np.savez_compressed(args.out, **save, symbols=np.array(sorted(out)), n_messages=nmsg)
log(f"wrote {args.out}")
