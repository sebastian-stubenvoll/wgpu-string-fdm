import numpy as np
from fractions import Fraction
from scipy.signal import butter, sosfiltfilt, resample_poly
from scipy.io import wavfile

TARGET_FS = 44100          
PEAK_TARGET = 0.89         
FADE_IN_S = 0.010          
FADE_OUT_S = 0.050         


def _highpass(x, fs, cutoff=20.0):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 30 or fs <= 2 * cutoff:
        return x - np.mean(x)
    sos = butter(4, cutoff / (0.5 * fs), btype="high", output="sos")
    return sosfiltfilt(sos, x)


def _fade(sig, fs, fade_in=FADE_IN_S, fade_out=FADE_OUT_S):
    n = sig.shape[0]
    fi = min(int(fade_in * fs), n // 4)
    fo = min(int(fade_out * fs), n // 4)
    env = np.ones(n)
    if fi > 0:
        env[:fi] = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, fi)))
    if fo > 0:
        env[-fo:] = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, fo)))
    return sig * env[:, None]


def _write_stereo_wav(path, stereo, fs_native):
    # 1. DC / sub-audible removal per channel
    y = np.stack(
        [_highpass(stereo[:, c], fs_native) for c in range(stereo.shape[1])],
        axis=1,
    )

    # 2. Resample to a standard rate
    frac = Fraction(int(round(TARGET_FS)), int(round(fs_native))).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator
    if up != down:
        y = resample_poly(y, up, down, axis=0)
    fs_out = int(round(fs_native * up / down))

    # 3. Joint peak normalisation (preserves the stereo image)
    peak_val = np.max(np.abs(y)) if y.size else 0.0
    if peak_val > 0:
        y = y * (PEAK_TARGET / peak_val)

    # 4. Fades guarantee silent, click-free endpoints
    y = _fade(y, fs_out)

    # 5. Safety clamp + int16 encode
    y = np.clip(y, -1.0, 1.0)
    pcm = (y * 32767.0).astype(np.int16)
    wavfile.write(path, fs_out, pcm)
    return path


def generate_node_audio(pos, vel, mom, fs_native, title_prefix="", filename_prefix="",
                        sources=("velocity", "displacement", "momentum")):
    data = {
        "velocity": np.asarray(vel, dtype=np.float64),
        "displacement": np.asarray(pos, dtype=np.float64),
        "momentum": np.asarray(mom, dtype=np.float64),
    }

    generated = []
    for name in sources:
        arr = data[name]
        if arr.ndim != 2 or arr.shape[1] < 3:
            continue
        stereo = np.stack([arr[:, 1], arr[:, 2]], axis=1)  # Y -> L, Z -> R
        path = f"{filename_prefix}_{name}.wav"
        try:
            _write_stereo_wav(path, stereo, fs_native)
            generated.append(path)
        except Exception as e:  
            print(f"[audio] Failed to render {name} for {title_prefix}: {e}")
    return generated
