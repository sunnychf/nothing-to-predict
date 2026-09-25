"""Pilot A: the synthetic-martingale probe (paper Section 4.1, plan stage 1).

Feeds a time series foundation model sequences that are unpredictable by
construction and measures what it forecasts anyway. Under the martingale null
the optimal forecast is persistence, so by Proposition 3.2 everything the model
adds on top of the last observed value is pure excess risk, and its magnitude is
the imported prior.

Stages (each appends to results/pilot_a.jsonl and skips configs already there,
because the box reboots without warning):

  ladder    the four rungs, per-horizon departure from the null
  floor     the artefact controls: Monte Carlo, codec, scale, random init
  spectral  long-horizon forecasts, is the departure patterned or white
  freq      declared-frequency attribution (see NOTE_ON_FREQ below)

Usage:
  python pilot_a.py --stage ladder --model amazon/chronos-t5-small
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np
import torch



from paths import RESULTS_DIR

RESULTS = os.environ.get("PILOT_A_OUT", os.path.join(RESULTS_DIR, "pilot_a.jsonl"))
CTX = 512          # Chronos t5 context length
DEFAULT_H = 16


# ------------------------------------------------------------------ plumbing

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def append(rec):
    os.makedirs(os.path.dirname(RESULTS) or ".", exist_ok=True)
    with open(RESULTS, "a") as f:
        f.write(json.dumps(rec, default=float) + "\n")
        f.flush()
        os.fsync(f.fileno())


def done_keys():
    if not os.path.exists(RESULTS):
        return set()
    ks = set()
    with open(RESULTS) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue          # a torn line from a reboot mid-write
            if "key" in r and not r.get("error"):
                ks.add(r["key"])
    return ks


# -------------------------------------------------------------------- models

def load_pipeline(model_id, init="pretrained", device=__import__("os").environ.get("TSFM_DEVICE","cuda"), dtype=torch.float32):
    """Load Chronos, either pretrained or with randomly initialised weights.

    The random-init variant keeps the architecture, config and tokeniser and
    throws away only what was learned, so it is the architecture's behaviour
    with no corpus behind it.
    """
    from chronos import ChronosPipeline, ChronosConfig, ChronosModel
    from transformers import AutoConfig, AutoModelForSeq2SeqLM

    if init == "pretrained":
        return ChronosPipeline.from_pretrained(model_id, device_map=device,
                                               torch_dtype=dtype)
    cfg = AutoConfig.from_pretrained(model_id)
    chronos_cfg = ChronosConfig(**cfg.chronos_config)
    torch.manual_seed(1234)
    inner = AutoModelForSeq2SeqLM.from_config(cfg).to(device=device, dtype=dtype)
    inner.eval()
    return ChronosPipeline(tokenizer=chronos_cfg.create_tokenizer(),
                           model=ChronosModel(config=chronos_cfg, model=inner))


@torch.no_grad()
def forecast(pipe, ctx_np, H, num_samples, batch=64, seed=0):
    """Sampled trajectories, shape (n_series, num_samples, H).

    Chronos expands the batch by num_samples internally, so the encoder sees
    batch * num_samples sequences at once and memory scales with the product,
    not with batch alone. This card is shared with the rest of the lab and its
    free memory moves during a run, so rather than pick a batch size that is
    safe today we halve on OOM and carry on. The result is identical; only the
    chunking changes.
    """
    torch.manual_seed(seed)
    outs = []
    i, bs = 0, max(1, min(batch, max(1, 1600 // max(1, num_samples))))
    while i < len(ctx_np):
        chunk = torch.tensor(ctx_np[i:i + bs], dtype=torch.float32)
        try:
            out = pipe.predict(chunk, prediction_length=H,
                               num_samples=num_samples,
                               limit_prediction_length=False)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs = max(1, bs // 2)
            log(f"  OOM -> batch {bs}")
            continue
        outs.append(out.float().cpu().numpy())
        i += bs
    return np.concatenate(outs, axis=0)


@torch.no_grad()
def codec_floor(pipe, ctx_np, y_last, H):
    """Lower bound on the departure imposed by the value codec alone.

    Chronos maps values to a finite token vocabulary after rescaling by the
    context. A model that wanted to emit exact persistence therefore CANNOT: the
    closest it can come to y_T is the round trip of y_T through that codec. We
    measure that round trip directly, which bounds below the departure of any
    forecast this pipeline can produce.

    This is a stricter and more honest control than a randomly initialised
    network, whose departure is dominated by its untrained output distribution
    rather than by the codec, and which therefore is not a floor at all. We run
    the random-init model too, but report it as an architecture reference.
    """
    target = np.repeat(np.asarray(y_last).reshape(-1, 1), H, axis=1)
    ctx_t = torch.tensor(ctx_np, dtype=torch.float32)
    tgt_t = torch.tensor(target, dtype=torch.float32)
    tok = pipe.tokenizer
    # Scale is fitted on the context exactly as at inference time, then the
    # would-be output is pushed through the same quantise/dequantise path.
    # We call the private _input_transform rather than label_input_transform
    # because the latter asserts the length equals config.prediction_length (64)
    # and appends an EOS token; neither belongs in a pure codec round trip.
    # output_transform already maps ids back by subtracting n_special_tokens + 1,
    # so the ids are handed to it unmodified.
    _, _, scale = tok.context_input_transform(ctx_t)
    ids, _, _ = tok._input_transform(tgt_t, scale)
    recon = tok.output_transform(ids.unsqueeze(1), scale).squeeze(1).numpy()
    return (recon - target) ** 2


from probe_metrics import make_data, summarise, spectra  # noqa: E402


# -------------------------------------------------------------------- stages

def stage_ladder(args, pipe, done):
    n, T, H = args.n, CTX, args.horizon
    for rung in ["N1", "N2", "N3_nu3", "N3_nu5", "N4"]:
        for seed in range(args.seeds):
            key = f"ladder|{args.model}|{args.init}|{rung}|s{seed}|H{H}|n{n}|ns{args.num_samples}"
            if key in done:
                log(f"skip {key}")
                continue
            y_ctx, y_true, info = make_data(rung, n, T, H, seed=1000 + seed)
            samples = forecast(pipe, y_ctx, H, args.num_samples, args.batch, seed)
            y_hat = samples.mean(axis=1)          # the L2-optimal point summary
            fl = codec_floor(pipe, y_ctx, y_ctx[:, -1], H)
            rec = {"key": key, "stage": "ladder", "model": args.model,
                   "init": args.init, "rung": rung, "seed": seed, "H": H,
                   "n": n, "num_samples": args.num_samples,
                   **summarise(y_hat, y_ctx, y_true, info, rung, H, floor=fl,
                               samples=samples),
                   **{f"spec_{k}": v for k, v in spectra(y_hat, y_ctx).items()}}
            # median as well, since a skewed sample cloud makes mean != median
            rec["imported_median_pt"] = np.mean(
                (np.median(samples, axis=1) - y_ctx[:, -1:]) ** 2, axis=0).tolist()
            append(rec)
            log(f"{rung} s{seed}: dep/sigma h1={rec['departure_in_sigma'][0]:.3f} "
                f"h{H}={rec['departure_in_sigma'][-1]:.3f} "
                f"floor_share_h1={rec['floor_share'][0]:.3f} "
                f"flat={rec['spec_flatness_detrended']:.3f}")


def stage_floor(args, pipe, done):
    """The three artefact controls of Section 4.1, plus the codec bound."""
    n, T, H = args.n_floor, CTX, args.horizon
    # 1. Monte Carlo: raise trajectories until the departure stops moving
    for ns in [5, 10, 20, 50, 100, 200, 500]:
        key = f"floor_mc|{args.model}|{args.init}|N1|ns{ns}|H{H}|n{n}"
        if key in done:
            log(f"skip {key}"); continue
        y_ctx, y_true, info = make_data("N1", n, T, H, seed=2000)
        s = forecast(pipe, y_ctx, H, ns, args.batch, 0)
        y_hat = s.mean(axis=1)
        fl = codec_floor(pipe, y_ctx, y_ctx[:, -1], H)
        append({"key": key, "stage": "floor_mc", "model": args.model,
                "init": args.init, "num_samples": ns, "H": H, "n": n,
                **summarise(y_hat, y_ctx, y_true, info, "N1", H, floor=fl,
                            samples=s)})
        log(f"mc ns={ns}: dep/sigma h1={np.sqrt(np.mean((y_hat[:,0]-y_ctx[:,-1])**2))/info['sigma']:.4f}")
    # 2. Scale: a quantisation artefact scales with the grid, a prior should not.
    #    We vary the step size relative to the level, which is what actually
    #    moves the signal against the token grid, and also vary the level itself
    #    to confirm the pipeline's scale equivariance separately.
    for sr in [1e-4, 1e-3, 1e-2, 1e-1]:
        key = f"floor_scale|{args.model}|{args.init}|N1|sr{sr:g}|H{H}|n{n}|ns{args.num_samples}"
        if key in done:
            log(f"skip {key}"); continue
        y_ctx, y_true, info = make_data("N1", n, T, H, seed=2001, sigma_rel=sr)
        s = forecast(pipe, y_ctx, H, args.num_samples, args.batch, 0)
        fl = codec_floor(pipe, y_ctx, y_ctx[:, -1], H)
        append({"key": key, "stage": "floor_scale", "model": args.model,
                "init": args.init, "sigma_rel": sr, "H": H, "n": n,
                "num_samples": args.num_samples,
                **summarise(s.mean(axis=1), y_ctx, y_true, info, "N1", H, floor=fl,
                            samples=s)})
        log(f"scale sr={sr:g}: done")
    for lv in [1.0, 100.0, 10000.0]:
        key = f"floor_level|{args.model}|{args.init}|N1|lv{lv:g}|H{H}|n{n}|ns{args.num_samples}"
        if key in done:
            log(f"skip {key}"); continue
        y_ctx, y_true, info = make_data("N1", n, T, H, seed=2002, level0=lv)
        s = forecast(pipe, y_ctx, H, args.num_samples, args.batch, 0)
        fl = codec_floor(pipe, y_ctx, y_ctx[:, -1], H)
        append({"key": key, "stage": "floor_level", "model": args.model,
                "init": args.init, "level0": lv, "H": H, "n": n,
                "num_samples": args.num_samples,
                **summarise(s.mean(axis=1), y_ctx, y_true, info, "N1", H, floor=fl,
                            samples=s)})
        log(f"level lv={lv:g}: done")


def stage_spectral(args, pipe, done):
    """Long horizon, because period 7 and 24 cannot be resolved in 16 steps."""
    n, T, H = args.n_spec, CTX, args.horizon_long
    for rung in ["N1", "N2"]:
        key = f"spectral|{args.model}|{args.init}|{rung}|H{H}|n{n}|ns{args.num_samples}"
        if key in done:
            log(f"skip {key}"); continue
        y_ctx, y_true, info = make_data(rung, n, T, H, seed=3000)
        s = forecast(pipe, y_ctx, H, args.num_samples, max(8, args.batch // 4), 0)
        y_hat = s.mean(axis=1)
        fl = codec_floor(pipe, y_ctx, y_ctx[:, -1], H)
        rec = {"key": key, "stage": "spectral", "model": args.model,
               "init": args.init, "rung": rung, "H": H, "n": n,
               "num_samples": args.num_samples,
               **summarise(y_hat, y_ctx, y_true, info, rung, H, floor=fl,
                           samples=s),
               **{f"spec_{k}": v for k, v in spectra(y_hat, y_ctx).items()}}
        append(rec)
        log(f"spectral {rung}: flat_detrended={rec['spec_flatness_detrended']:.4f} "
            f"peak_period={rec['spec_peak_period']:.1f} "
            f"band24={rec['spec_band_power']['p24']:.4f} "
            f"band7={rec['spec_band_power']['p7']:.4f}")


def stage_freq(args, pipe, done):
    """Declared-frequency attribution.

    NOTE_ON_FREQ: Chronos takes only a tensor of values. It is given no
    timestamps, no frequency string and no calendar features, so there is
    nothing to vary and this test is not defined for it. We record that fact
    rather than fabricating a variation, and the test moves to a model that does
    accept a declaration (TimesFM's freq index, Moirai, Time-MoE).
    """
    sig = None
    try:
        import inspect
        sig = str(inspect.signature(pipe.predict))
    except (TypeError, ValueError):
        pass
    key = f"freq|{args.model}|{args.init}|not-applicable"
    if key in done:
        log(f"skip {key}"); return
    append({"key": key, "stage": "freq", "model": args.model, "init": args.init,
            "applicable": False, "predict_signature": sig,
            "reason": "Chronos conditions on values only; no frequency is declared "
                      "to it, so the declared-frequency test is undefined for this "
                      "model and must be run on TimesFM / Moirai / Time-MoE."})
    log(f"freq: not applicable for {args.model}; signature = {sig}")


STAGES = {"ladder": stage_ladder, "floor": stage_floor,
          "spectral": stage_spectral, "freq": stage_freq}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="ladder", choices=list(STAGES) + ["all"])
    ap.add_argument("--model", default="amazon/chronos-t5-small")
    ap.add_argument("--init", default="pretrained", choices=["pretrained", "random"])
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--n-floor", type=int, default=256)
    ap.add_argument("--n-spec", type=int, default=128)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=DEFAULT_H)
    ap.add_argument("--horizon-long", type=int, default=128)
    ap.add_argument("--num-samples", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default=__import__("os").environ.get("TSFM_DEVICE","cuda"))
    args = ap.parse_args()

    log(f"loading {args.model} init={args.init} on {args.device}")
    pipe = load_pipeline(args.model, args.init, args.device)
    done = done_keys()
    log(f"{len(done)} configs already done")

    stages = list(STAGES) if args.stage == "all" else [args.stage]
    failed = False
    for st in stages:
        log(f"===== stage {st} =====")
        try:
            STAGES[st](args, pipe, done)
        except Exception:
            failed = True
            tb = traceback.format_exc()
            # Make the failure louder than any success in the log.
            print("!" * 70, flush=True)
            print(f"STAGE {st} FAILED", flush=True)
            print(tb, flush=True)
            print("!" * 70, flush=True)
            append({"key": f"ERROR|{st}|{args.model}|{args.init}|{time.time()}",
                    "stage": st, "error": True, "traceback": tb})
    log("DONE" if not failed else "DONE WITH FAILURES")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
