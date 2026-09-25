"""One forecaster interface for the five model families, reusing each Pilot A driver's
loader and forecast function unchanged, so a window is forecast exactly as in the
ladder runs (same defaults: Chronos S=100; Moirai patch auto, S=100; TimesFM freq 0,
mean channel; FinCast freq 0, mean mode; Time-MoE greedy multi-horizon decode).

    fn = get_forecaster(model, H, samples=100, batch=64)
    yhat, mcvar = fn(ctx)          # ctx (n, T) float -> (n, H) point forecast, (n, H) Var(samples)/S

mcvar is zero for the point-forecast models. Each model must be called inside its own
environment (env.sh / env_moirai.sh / env_timesfm.sh / env_timemoe.sh / env_fincast.sh).
"""
import numpy as np

SAMPLED = ("chronos", "moirai", "sundial")


def get_forecaster(model, H, samples=100, batch=64, ctx_len=None):
    """ctx_len: context length for the models that fix it at model-build time (Moirai); the others
    take the input's length as it comes, up to their own maximum (Chronos-T5 512, Bolt and
    TimesFM 2048, Time-MoE 4096, Chronos-2 8192, Sundial 2880). Default None = the paper's 512."""
    if model == "chronos":
        from pilot_a import load_pipeline, forecast
        pipe = load_pipeline("amazon/chronos-t5-small")

        def fn(c):
            s = forecast(pipe, c, H, samples, batch=batch, seed=0)      # (n, S, H)
            return s.mean(1), s.var(1) / s.shape[1]
    elif model == "fincast":
        from pilot_a_fincast import load_fincast, forecast_fincast
        api = load_fincast(H)

        def fn(c):
            m, _ = forecast_fincast(api, c, H, 0, batch); return m, np.zeros_like(m)
    elif model == "timesfm":
        from pilot_a_timesfm import load_timesfm, forecast_tfm
        tfm = load_timesfm(H)

        def fn(c):
            m, _ = forecast_tfm(tfm, c, H, 0, batch); return m, np.zeros_like(m)
    elif model == "timemoe":
        from pilot_a_timemoe import load_model, forecast_tm
        mdl = load_model()

        def fn(c):
            m = forecast_tm(mdl, c, H, batch); return m, np.zeros_like(m)
    elif model in ("chronosbolt", "chronos2"):
        from pilot_a_chronosx import load_pipe, forecast_cx, MODELS
        pipe = load_pipe(MODELS["bolt" if model == "chronosbolt" else "chronos2"][0])

        def fn(c):
            m, _ = forecast_cx(pipe, c, H, batch); return m, np.zeros_like(m)
    elif model == "tirex":
        from pilot_a_tirex import load_tirex, forecast_tirex
        mdl = load_tirex()

        def fn(c):
            m, _ = forecast_tirex(mdl, c, H, batch); return m, np.zeros_like(m)
    elif model == "moirai2":
        from pilot_a_moirai2 import load_moirai2, forecast_moirai2
        mdl = load_moirai2()

        def fn(c):
            m, _ = forecast_moirai2(mdl, c, H, batch); return m, np.zeros_like(m)
    elif model == "timesfm25":
        from pilot_a_timesfm25 import load_timesfm25, forecast_timesfm25
        mdl = load_timesfm25(flip=True, batch=batch)                    # the model as released (flip invariance on)

        def fn(c):
            m, _ = forecast_timesfm25(mdl, c, H, batch); return m, np.zeros_like(m)
    elif model == "sundial":
        from pilot_a_sundial import load_sundial, forecast_sundial
        mdl = load_sundial()

        def fn(c):
            s = forecast_sundial(mdl, c, H, samples, batch, seed=0)      # (n, S, H)
            return s.mean(1), s.var(1) / s.shape[1]
    elif model == "moirai":
        from pilot_a_moirai import forecast_moirai

        def fn(c):
            s = forecast_moirai(c, H, samples, "auto", batch=16, seed=0, ctx_len=ctx_len)      # (n, S, H)
            return s.mean(1), s.var(1) / s.shape[1]
    else:
        raise ValueError(model)
    return fn
