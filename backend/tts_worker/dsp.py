"""Broadcast-style post-processing for commentary clips (numpy only).

master(x, sr, target_lufs) runs:  high-pass + presence EQ -> loudness normalise
-> gentle compression -> loudness normalise -> peak limiter (-1 dBFS).

Loudness is ITU-R BS.1770 integrated loudness (K-weighting applied as a
zero-phase magnitude response in the FFT domain, 400 ms blocks, 75 % overlap,
absolute -70 LUFS and relative -10 LU gates). Equivalent to pyloudnorm for
energy purposes, without needing scipy.
"""
from __future__ import annotations

import io
import math
import wave
from typing import List, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------
# basic helpers


def db_to_lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def silence(ms: float, sr: int) -> np.ndarray:
    return np.zeros(int(round(sr * ms / 1000.0)), dtype=np.float32)


def fade(x: np.ndarray, sr: int, in_ms: float = 6.0, out_ms: float = 10.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).copy()
    n_in = min(len(x) // 2, int(sr * in_ms / 1000))
    n_out = min(len(x) // 2, int(sr * out_ms / 1000))
    if n_in > 0:
        x[:n_in] *= np.linspace(0.0, 1.0, n_in, dtype=np.float32)
    if n_out > 0:
        x[-n_out:] *= np.linspace(1.0, 0.0, n_out, dtype=np.float32)
    return x


def _frame_rms_db(x: np.ndarray, sr: int, frame_ms: float = 10.0) -> np.ndarray:
    n = max(1, int(sr * frame_ms / 1000))
    usable = len(x) // n * n
    if usable == 0:
        return np.array([-120.0])
    fr = x[:usable].reshape(-1, n)
    return 10.0 * np.log10(np.mean(fr.astype(np.float64) ** 2, axis=1) + 1e-12)


def trim_silence(x: np.ndarray, sr: int, thresh_db: float = -42.0, pad_ms: float = 15.0) -> np.ndarray:
    """Trim leading/trailing silence relative to the clip's loudest 10 ms frame."""
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0:
        return x
    db = _frame_rms_db(x, sr, 10.0)
    loud = np.where(db > db.max() + thresh_db)[0]
    if len(loud) == 0:
        return x
    n = int(sr * 0.010)
    pad = int(sr * pad_ms / 1000)
    a = max(0, loud[0] * n - pad)
    b = min(len(x), (loud[-1] + 1) * n + pad)
    return x[a:b]


def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    """Band-limited FFT resampling (fine for short clips)."""
    x = np.asarray(x, dtype=np.float32)
    if sr_from == sr_to or len(x) == 0:
        return x
    n_out = int(round(len(x) * sr_to / sr_from))
    pad = int(0.05 * sr_from)  # reflect-pad to tame edge ringing
    xp = np.concatenate([np.zeros(pad, np.float32), x, np.zeros(pad, np.float32)])
    n_in_p = len(xp)
    n_out_p = int(round(n_in_p * sr_to / sr_from))
    X = np.fft.rfft(xp)
    m = n_out_p // 2 + 1
    if m <= len(X):
        Y = X[:m].copy()
    else:
        Y = np.zeros(m, dtype=X.dtype)
        Y[:len(X)] = X
    y = np.fft.irfft(Y, n_out_p) * (n_out_p / n_in_p)
    p_out = int(round(pad * sr_to / sr_from))
    return y[p_out:p_out + n_out].astype(np.float32)


def pitch_shift(x: np.ndarray, semitones: float) -> np.ndarray:
    """Shift pitch (and formants) by resampling. Changes duration by 2^(-st/12);
    callers pre-compensate with the synthesis speed."""
    if abs(semitones) < 1e-3:
        return np.asarray(x, dtype=np.float32)
    r = 2.0 ** (semitones / 12.0)
    n = len(x)
    return resample(x, int(round(48000 * r)), 48000)[: int(round(n / r))]


# --------------------------------------------------------------------------
# frequency-domain (zero-phase) filtering with RBJ biquad magnitude responses


def _biquad(kind: str, fc: float, sr: int, q: float = 0.707, gain_db: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * fc / sr
    cw, sw = math.cos(w0), math.sin(w0)
    alpha = sw / (2.0 * q)
    if kind == "highpass":
        b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "peak":
        b = [1 + alpha * A, -2 * cw, 1 - alpha * A]
        a = [1 + alpha / A, -2 * cw, 1 - alpha / A]
    elif kind == "highshelf":
        sa = 2 * math.sqrt(A) * alpha
        b = [A * ((A + 1) + (A - 1) * cw + sa), -2 * A * ((A - 1) + (A + 1) * cw), A * ((A + 1) + (A - 1) * cw - sa)]
        a = [(A + 1) - (A - 1) * cw + sa, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - sa]
    elif kind == "lowshelf":
        sa = 2 * math.sqrt(A) * alpha
        b = [A * ((A + 1) - (A - 1) * cw + sa), 2 * A * ((A - 1) - (A + 1) * cw), A * ((A + 1) - (A - 1) * cw - sa)]
        a = [(A + 1) + (A - 1) * cw + sa, -2 * ((A - 1) + (A + 1) * cw), (A + 1) + (A - 1) * cw - sa]
    else:
        raise ValueError(kind)
    return np.array(b, dtype=np.float64), np.array(a, dtype=np.float64)


def _mag(b: np.ndarray, a: np.ndarray, n_fft: int) -> np.ndarray:
    w = np.linspace(0.0, math.pi, n_fft // 2 + 1)
    z = np.exp(-1j * w)
    num = b[0] + b[1] * z + b[2] * z * z
    den = a[0] + a[1] * z + a[2] * z * z
    return np.abs(num / den)


def _apply_mag(x: np.ndarray, mag_fn) -> np.ndarray:
    n = len(x)
    if n == 0:
        return x
    n_fft = 1 << int(math.ceil(math.log2(n + 2048)))
    X = np.fft.rfft(x.astype(np.float64), n_fft)
    y = np.fft.irfft(X * mag_fn(n_fft), n_fft)[:n]
    return y.astype(np.float32)


def eq_broadcast(x: np.ndarray, sr: int, presence_db: float = 2.5) -> np.ndarray:
    """High-pass 80 Hz, small low-mid dip, presence lift ~3.2 kHz, a little air.
    Helps a voice cut through a crowd bed without sounding harsh."""
    stages = [
        _biquad("highpass", 80.0, sr, q=0.707),
        _biquad("peak", 300.0, sr, q=1.0, gain_db=-1.5),
        _biquad("peak", 3200.0, sr, q=0.9, gain_db=presence_db),
    ]
    if sr > 18000:
        stages.append(_biquad("highshelf", 8000.0, sr, q=0.707, gain_db=1.2))

    def mag(n_fft: int) -> np.ndarray:
        m = np.ones(n_fft // 2 + 1)
        for b, a in stages:
            m *= _mag(b, a, n_fft)
        return m

    return _apply_mag(x, mag)


# --------------------------------------------------------------------------
# loudness (BS.1770)


def _k_weight(x: np.ndarray, sr: int) -> np.ndarray:
    # pyloudnorm's BS.1770 stage 1 (high shelf +4 dB @1.5 kHz) and stage 2 (RLB high-pass 38 Hz)
    s1 = _biquad("highshelf", 1500.0, sr, q=1 / math.sqrt(2), gain_db=4.0)
    s2 = _biquad("highpass", 38.0, sr, q=0.5)
    return _apply_mag(x, lambda n: _mag(*s1, n) * _mag(*s2, n))


def loudness_lufs(x: np.ndarray, sr: int) -> float:
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0 or not np.any(x):
        return -120.0
    y = _k_weight(x, sr).astype(np.float64)
    blk = int(0.400 * sr)
    hop = int(0.100 * sr)
    if len(y) < blk:
        z = [np.mean(y ** 2)]
    else:
        c = np.concatenate([[0.0], np.cumsum(y ** 2)])
        starts = np.arange(0, len(y) - blk + 1, hop)
        z = (c[starts + blk] - c[starts]) / blk
    z = np.asarray(z)
    lk = -0.691 + 10 * np.log10(z + 1e-15)
    z = z[lk > -70.0]
    if len(z) == 0:
        return -120.0
    rel = -0.691 + 10 * np.log10(np.mean(z)) - 10.0
    lk = -0.691 + 10 * np.log10(z + 1e-15)
    z2 = z[lk > rel]
    if len(z2) == 0:
        z2 = z
    return float(-0.691 + 10 * np.log10(np.mean(z2)))


def normalize_lufs(x: np.ndarray, sr: int, target: float = -16.0, max_gain_db: float = 30.0) -> np.ndarray:
    L = loudness_lufs(x, sr)
    if L <= -100:
        return np.asarray(x, dtype=np.float32)
    g = max(-30.0, min(max_gain_db, target - L))
    return (np.asarray(x, dtype=np.float32) * db_to_lin(g)).astype(np.float32)


# --------------------------------------------------------------------------
# dynamics


def compress(x: np.ndarray, sr: int, threshold_db: float = -20.0, ratio: float = 2.5,
             knee_db: float = 6.0, attack_ms: float = 8.0, release_ms: float = 120.0) -> np.ndarray:
    """Feed-forward RMS compressor on 5 ms frames, gain interpolated per sample."""
    x = np.asarray(x, dtype=np.float32)
    n = max(1, int(sr * 0.005))
    frames = len(x) // n + 1
    xp = np.zeros(frames * n, dtype=np.float32)
    xp[:len(x)] = x
    lvl = 10.0 * np.log10(np.mean(xp.reshape(frames, n).astype(np.float64) ** 2, axis=1) + 1e-12)
    over = lvl - threshold_db
    gr = np.zeros_like(over)
    hk = knee_db / 2.0
    above = over > hk
    inknee = (over > -hk) & ~above
    gr[above] = over[above] * (1.0 - 1.0 / ratio)
    gr[inknee] = (1.0 - 1.0 / ratio) * (over[inknee] + hk) ** 2 / (2.0 * knee_db)
    a_att = math.exp(-5.0 / max(attack_ms, 0.1))
    a_rel = math.exp(-5.0 / max(release_ms, 0.1))
    sm = np.empty_like(gr)
    g = 0.0
    for i, t in enumerate(gr):          # a few hundred frames per clip
        coef = a_att if t > g else a_rel
        g = coef * g + (1.0 - coef) * t
        sm[i] = g
    centers = np.arange(frames) * n + n / 2.0
    gain_db = np.interp(np.arange(len(x)), centers, -sm)
    return (x * (10.0 ** (gain_db / 20.0))).astype(np.float32)


def limit(x: np.ndarray, sr: int, ceiling_db: float = -1.0, lookahead_ms: float = 3.0,
          release_ms: float = 60.0) -> np.ndarray:
    """Look-ahead peak limiter: never lets |x| exceed the ceiling."""
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0:
        return x
    ceil = db_to_lin(ceiling_db)
    a = np.abs(x)
    if a.max() <= ceil:
        return x
    need = np.minimum(1.0, ceil / np.maximum(a, 1e-9))
    la = max(1, int(sr * lookahead_ms / 1000))
    # running minimum over the look-ahead window (window centred on the sample)
    padded = np.concatenate([np.ones(la), need, np.ones(la)])
    from numpy.lib.stride_tricks import sliding_window_view
    gmin = sliding_window_view(padded, 2 * la + 1).min(axis=1)[: len(x)]
    # smooth the release (moving average keeps it below the requirement after a min-filter)
    k = max(1, int(sr * release_ms / 1000 / 4))
    c = np.concatenate([[0.0], np.cumsum(np.concatenate([np.ones(k) * gmin[0], gmin]))])
    smooth = (c[k:] - c[:-k])[: len(x)] / k
    g = np.minimum(gmin, smooth)
    y = x * g
    return np.clip(y, -ceil, ceil).astype(np.float32)


def master(x: np.ndarray, sr: int, target_lufs: float = -16.0, presence_db: float = 2.5) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0:
        return x
    x = eq_broadcast(x, sr, presence_db=presence_db)
    x = normalize_lufs(x, sr, target_lufs)
    x = compress(x, sr, threshold_db=target_lufs - 4.0, ratio=2.5)
    x = normalize_lufs(x, sr, target_lufs)
    x = limit(x, sr, ceiling_db=-1.0)
    return x


# --------------------------------------------------------------------------
# encoding

_OPUS_OK: Optional[bool] = None


def opus_available() -> bool:
    """True when soundfile/libsndfile can write *and read back* OGG/Opus."""
    global _OPUS_OK
    if _OPUS_OK is None:
        try:
            import soundfile as sf  # type: ignore
            t = np.sin(np.linspace(0, 2 * math.pi * 220, 12000)).astype(np.float32) * 0.3
            buf = io.BytesIO()
            sf.write(buf, t, 24000, format="OGG", subtype="OPUS")
            buf.seek(0)
            back, sr = sf.read(buf, dtype="float32")
            _OPUS_OK = bool(sr == 24000 and len(back) > 10000)
        except Exception:
            _OPUS_OK = False
    return _OPUS_OK


def encode(x: np.ndarray, sr: int, prefer: str = "ogg") -> Tuple[bytes, str]:
    """(bytes, mime). OGG/Opus at 24/48 kHz when available, else 16-bit WAV."""
    x = np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0)
    if prefer == "ogg" and sr in (8000, 12000, 16000, 24000, 48000) and opus_available():
        try:
            import soundfile as sf  # type: ignore
            buf = io.BytesIO()
            sf.write(buf, x, sr, format="OGG", subtype="OPUS")
            return buf.getvalue(), "audio/ogg"
        except Exception:
            pass
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes((x * 32767.0).astype("<i2").tobytes())
    return buf.getvalue(), "audio/wav"


# --------------------------------------------------------------------------
# analysis helpers (used by tests / diagnostics)


def silence_segments(x: np.ndarray, sr: int, thresh_db: float = -45.0, min_ms: float = 60.0) -> List[Tuple[float, float]]:
    """[(start_s, dur_ms)] of internal silences (relative to the loudest frame)."""
    db = _frame_rms_db(np.asarray(x, dtype=np.float32), sr, 10.0)
    quiet = db < db.max() + thresh_db
    out, i, n = [], 0, len(quiet)
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]:
                j += 1
            if i > 0 and j < n and (j - i) * 10 >= min_ms:
                out.append((i * 0.010, (j - i) * 10.0))
            i = j
        else:
            i += 1
    return out


def pitch_hz(x: np.ndarray, sr: int, fmin: float = 60.0, fmax: float = 400.0) -> float:
    """Median F0 over voiced 40 ms frames via normalised autocorrelation."""
    x = np.asarray(x, dtype=np.float64)
    n = int(0.040 * sr)
    hop = int(0.020 * sr)
    lo, hi = int(sr / fmax), int(sr / fmin)
    f0s = []
    db = _frame_rms_db(x.astype(np.float32), sr, 40.0)
    thr = db.max() - 25.0
    for k, s in enumerate(range(0, len(x) - n - hi, hop)):
        fr = x[s:s + n + hi]
        e = fr[:n]
        if 10 * np.log10(np.mean(e ** 2) + 1e-12) < thr:
            continue
        e = e - e.mean()
        best, best_lag = 0.0, 0
        denom0 = np.sqrt(np.sum(e * e)) + 1e-12
        for lag in range(lo, hi):
            seg = fr[lag:lag + n] - fr[lag:lag + n].mean()
            r = float(np.dot(e, seg) / (denom0 * (np.sqrt(np.sum(seg * seg)) + 1e-12)))
            if r > best:
                best, best_lag = r, lag
        if best > 0.6 and best_lag:
            f0s.append(sr / best_lag)
    return float(np.median(f0s)) if f0s else 0.0
