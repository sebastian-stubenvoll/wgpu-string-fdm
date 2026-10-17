"""Model-free dispersion analysis, streamed chunk-by-chunk.

Two independent measurements of (k, w) at every resonance of the string:

1. k-w spectrum: windowed FFT along x over the interior of the string (no
   boundary assumption, only that the interior is uniform), then a complex
   Welch FFT over time. k at a resonance = peak of that frequency row.
2. Mode-shape fit: the measured operating shape at each resonance is rebuilt
   in physical space and A sin(kx) + B cos(kx) is fitted to its interior, so k
   is a fitted parameter (precise even for the lowest modes, where the k-w
   map is resolution-limited to ~2pi/L).

The sine-series (DST) coefficients are used only as lossless compression of
the displacement field for (2): the end nodes are held at exactly zero by the
simulation, so the field is fully represented by them up to the kept band.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.fft import dst, fft, rfft, rfftfreq
from scipy.signal import get_window, find_peaks


# --------------------------------------------------------------------------
# Per-chunk spatial transforms (run in the extraction workers)
# --------------------------------------------------------------------------

def modal_projection(pos, n_modes=256):
    """Sine-series coefficients of displacement frames.

    pos: (frames, nodes, 3). Returns (frames, n_modes, 3) float32 with
    u_j = sum_n a_n sin(n*pi*j/(N-1)) exactly for band-limited fields.
    """
    pos = np.asarray(pos, dtype=np.float32)
    N = pos.shape[1]
    n_modes = min(n_modes, N - 2)
    coeffs = dst(pos[:, 1:-1, :], type=1, axis=1)[:, :n_modes, :]
    return (coeffs / np.float32(N - 1)).astype(np.float32)


def kspace_grid(node_count, spacing, k_max, margin=0.05, pad=2):
    """Interior slice, window, FFT length and kept k bins for kspace_projection."""
    i0 = max(1, int(np.ceil(margin * (node_count - 1))))
    i1 = node_count - i0
    m = i1 - i0
    nfft = pad * (1 << int(np.ceil(np.log2(m))))
    k = 2 * np.pi * np.fft.fftfreq(nfft, d=spacing)[:nfft // 2 + 1]
    n_k = int(min(k.size, np.searchsorted(k, k_max) + 1))
    window = get_window('hann', m).astype(np.float32)
    return {"i0": i0, "i1": i1, "nfft": nfft, "k": k[:n_k], "window": window}


def kspace_projection(pos, grid):
    """Windowed spatial FFT over the string interior.

    Returns (frames, n_k, 3) complex64 for k >= 0 (the field is real, so
    negative k carries no extra information once time is transformed with a
    full complex FFT).
    """
    seg = np.asarray(pos[:, grid["i0"]:grid["i1"], :], dtype=np.float32)
    seg = seg * grid["window"][None, :, None]
    spec = fft(seg, n=grid["nfft"], axis=1)[:, :grid["k"].size, :]
    return spec.astype(np.complex64)


# --------------------------------------------------------------------------
# Streaming Welch accumulators (run in the main process, chunks in order)
# --------------------------------------------------------------------------

class _StreamingWelch:
    """Rolling one-segment buffer with 50% overlap; subclasses accumulate."""

    def __init__(self, fs, nperseg=8192):
        self.fs = float(fs)
        self.nperseg = int(nperseg)
        self.step = self.nperseg // 2
        self.buf = None
        self.fill = 0
        self.n_segments = 0

    def _segment(self, seg, nfft):
        raise NotImplementedError

    def _prepare(self, seg):
        win = get_window('hann', seg.shape[0]).astype(np.float32)
        seg = seg - seg.mean(axis=0, keepdims=True)
        scale = 1.0 / (self.fs * np.sum(win.astype(np.float64) ** 2))
        return seg * win[:, None, None], scale

    def feed(self, chunk):
        """chunk: (frames, ..., 3), consecutive to the previous one."""
        if self.buf is None:
            self.buf = np.zeros((self.nperseg,) + chunk.shape[1:], dtype=chunk.dtype)
        i = 0
        while i < chunk.shape[0]:
            take = min(self.nperseg - self.fill, chunk.shape[0] - i)
            self.buf[self.fill:self.fill + take] = chunk[i:i + take]
            self.fill += take
            i += take
            if self.fill == self.nperseg:
                self._segment(self.buf, self.nperseg)
                self.n_segments += 1
                self.buf[:self.nperseg - self.step] = self.buf[self.step:]
                self.fill = self.nperseg - self.step

    def _flush(self):
        """Run shorter than one segment: one zero-padded segment."""
        if self.n_segments == 0 and self.buf is not None and self.fill >= 4:
            self._segment(self.buf[:self.fill], self.nperseg)
            self.n_segments += 1
        return self.n_segments > 0


class StreamingModalSpectra(_StreamingWelch):
    """Per-mode PSDs plus cross-spectra against a fixed random reference.

    The reference is a random-weighted sum over all sine coefficients, i.e. a
    random-weighted sum of displacement over the whole string, so it cannot
    sit on a node of any mode. The cross-spectrum C_n(f) = <A_n(f) R*(f)> is a
    phase-consistent (averageable) estimate of the operating shape at f.
    """

    def __init__(self, fs, nperseg=8192, seed=1234):
        super().__init__(fs, nperseg)
        self.seed = seed
        self.ref_w = None
        self.psd = None     # (n_modes, F, 3) float64
        self.cross = None   # (F, n_modes, 3) complex128

    def _segment(self, seg, nfft):
        seg, scale = self._prepare(np.asarray(seg, dtype=np.float32))
        spec = rfft(seg, n=nfft, axis=0)                     # (F, n_modes, 3)
        if self.ref_w is None:
            self.ref_w = np.random.default_rng(self.seed).standard_normal(spec.shape[1]).astype(np.float32)
        # Per-segment temporaries stay single precision; only the running
        # sums are double, which keeps the main process's peak memory low.
        power = (spec.real ** 2 + spec.imag ** 2).transpose(1, 0, 2)
        ref = np.einsum('fnc,n->fc', spec, self.ref_w)       # (F, 3)
        cross = spec * np.conj(ref)[:, None, :]
        if self.psd is None:
            self.psd = np.zeros(power.shape, np.float64)
            self.cross = np.zeros(cross.shape, np.complex128)
        self.psd += scale * power
        self.cross += scale * cross

    def finalize(self):
        if not self._flush():
            return None
        return {
            "freqs": rfftfreq(self.nperseg, 1.0 / self.fs),
            "psd_trans": (self.psd[:, :, 1] + self.psd[:, :, 2]) / self.n_segments,
            "psd_long": self.psd[:, :, 0] / self.n_segments,
            "cross_trans": ((self.cross[:, :, 1] + self.cross[:, :, 2]) / self.n_segments).astype(np.complex64),
            "cross_long": (self.cross[:, :, 0] / self.n_segments).astype(np.complex64),
            "n_segments": self.n_segments,
        }


class StreamingKOmegaSpectra(_StreamingWelch):
    """|U(k, w)|^2 from complex k-space series; +w and -w (the two travel
    directions) are folded together onto w >= 0."""

    def __init__(self, fs, nperseg=8192):
        super().__init__(fs, nperseg)
        self.psd = None     # (F_pos, n_k, 3) float64

    def _segment(self, seg, nfft):
        seg, scale = self._prepare(np.asarray(seg, dtype=np.complex64))
        spec = fft(seg, n=nfft, axis=0)                      # (nfft, n_k, 3)
        power = spec.real ** 2 + spec.imag ** 2
        del spec
        half = nfft // 2
        if self.psd is None:
            self.psd = np.zeros((half + 1,) + power.shape[1:], np.float64)
        self.psd += scale * power[:half + 1]
        self.psd[1:half] += scale * power[nfft - 1:half:-1]

    def finalize(self):
        if not self._flush():
            return None
        return {
            "kw_trans": ((self.psd[:, :, 1] + self.psd[:, :, 2]) / self.n_segments).astype(np.float32),
            "kw_long": (self.psd[:, :, 0] / self.n_segments).astype(np.float32),
        }


class StreamingCrossSpectrum(_StreamingWelch):
    """Time-averaged product of two coefficient series per mode, resolved in
    frequency: Re<A_n*(f) B_n(f)>, one-sided, as a density (per Hz) whose
    integral over f is the Hann^2-weighted segment mean of a_n(t) b_n(t)
    (segment means removed). Fed (frames, n_modes, 2)."""

    def __init__(self, fs, nperseg=8192):
        super().__init__(fs, nperseg)
        self.acc = None     # (n_modes, F) float64

    def _segment(self, seg, nfft):
        seg, scale = self._prepare(np.asarray(seg, dtype=np.float32))
        spec = rfft(seg, n=nfft, axis=0)                     # (F, n_modes, 2)
        cross = (np.conj(spec[..., 0]) * spec[..., 1]).real.astype(np.float64)
        del spec
        cross[1:(None if nfft % 2 else -1)] *= 2.0           # fold negative frequencies
        if self.acc is None:
            self.acc = np.zeros(cross.T.shape, np.float64)
        self.acc += scale * cross.T

    def finalize(self):
        if not self._flush():
            return None
        return {"freqs": rfftfreq(self.nperseg, 1.0 / self.fs), "p_nf": self.acc / self.n_segments,
                "n_segments": self.n_segments}


def finalize_dispersion(modal_acc, kw_acc, k_grid, node_count, length):
    """Combine both accumulators into one picklable dict for plot_dispersion."""
    modal = modal_acc.finalize()
    kw = kw_acc.finalize()
    if modal is None or kw is None:
        return None
    return {**modal, **kw, "k_grid": np.asarray(k_grid), "node_count": int(node_count),
            "length": float(length)}


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------

def _parabolic_offset(lm, l0, lp):
    """Sub-bin offset of a peak from log-values at bins i-1, i, i+1."""
    denom = lm - 2.0 * l0 + lp
    with np.errstate(divide='ignore', invalid='ignore'):
        d = np.where(denom < 0, 0.5 * (lm - lp) / denom, 0.0)
    return np.clip(d, -0.5, 0.5)


def find_resonances(freqs, psd_modes, prominence_db=6.0, floor_db=60.0):
    """Resonance frequencies from the spatially summed power spectrum.

    sum_n |A_n(f)|^2 is proportional to the integral of |U(x,f)|^2 over the
    string (Parseval), so this uses no assumption about mode shapes.
    Returns (bin indices, interpolated frequencies in Hz).
    """
    total = psd_modes.sum(axis=0)
    if total.max() <= 0:
        return np.array([], int), np.array([])
    db = 10.0 * np.log10(np.maximum(total, total.max() * 1e-12))
    idx, _ = find_peaks(db, prominence=prominence_db, height=db.max() - floor_db)
    idx = idx[(idx > 0) & (idx < db.size - 1)]
    off = _parabolic_offset(db[idx - 1], db[idx], db[idx + 1])
    return idx, freqs[idx] + off * (freqs[1] - freqs[0])


def k_from_kw_rows(kw, k_grid, rows):
    """Peak k of each k-w frequency row (sub-bin interpolated)."""
    out = np.full(rows.size, np.nan)
    tiny = np.finfo(np.float64).tiny
    for j, r in enumerate(rows):
        row = np.log(np.maximum(kw[r].astype(np.float64), tiny))
        i = int(np.argmax(row))
        if 0 < i < row.size - 1:
            out[j] = k_grid[i] + _parabolic_offset(row[i - 1], row[i], row[i + 1]) * (k_grid[1] - k_grid[0])
        else:
            out[j] = k_grid[i]
    return out


def _fit_k(x, u, k_lo, k_hi, n_grid=600):
    """Least-squares k for u ~ A sin(kx) + B cos(kx) (A, B solved exactly for
    each k). Returns (k, R^2)."""
    def rss(ks):
        s = np.sin(np.outer(ks, x))
        c = np.cos(np.outer(ks, x))
        a, b, cc = (s * s).sum(1), (s * c).sum(1), (c * c).sum(1)
        ys, yc = s @ u, c @ u
        det = a * cc - b * b
        det = np.where(np.abs(det) < 1e-30, 1e-30, det)
        A = (ys * cc - yc * b) / det
        B = (yc * a - ys * b) / det
        return np.sum(u * u) - (A * ys + B * yc)

    ks = np.linspace(k_lo, k_hi, n_grid)
    r = rss(ks)
    i = int(np.argmin(r))
    # Refine around the best grid point.
    fine = np.linspace(ks[max(i - 1, 0)], ks[min(i + 1, n_grid - 1)], 200)
    rf = rss(fine)
    j = int(np.argmin(rf))
    total = np.sum(u * u)
    return fine[j], (1.0 - rf[j] / total) if total > 0 else 0.0


def k_from_mode_shapes(cross, rows, node_count, length, margin=0.05, min_r2=0.95):
    """Fit k to the interior of the measured operating shape at each row.

    Returns (k, R^2, accepted mask).
    """
    n_modes = cross.shape[1]
    j = np.arange(node_count)
    x = j * length / (node_count - 1)
    basis = np.sin(np.outer(np.arange(1, n_modes + 1), np.pi * j / (node_count - 1)))  # (n, N)
    i0 = max(1, int(np.ceil(margin * (node_count - 1))))
    xi = x[i0:node_count - i0]
    L_int = xi[-1] - xi[0]
    dk = 2 * np.pi / L_int

    ks, r2s = np.full(rows.size, np.nan), np.zeros(rows.size)
    for m, r in enumerate(rows):
        shape = (cross[r].astype(np.complex128) @ basis)[i0:node_count - i0]
        # Standing shape: rotate the common phase away and keep the real part.
        theta = 0.5 * np.angle(np.sum(shape * shape))
        u = np.real(shape * np.exp(-1j * theta))
        # Initial guess from a zero-padded FFT of the interior shape.
        nfft = 16 * (1 << int(np.ceil(np.log2(u.size))))
        spec = np.abs(np.fft.rfft(u * np.hanning(u.size), n=nfft))
        k0 = 2 * np.pi * np.fft.rfftfreq(nfft, d=xi[1] - xi[0])[np.argmax(spec)]
        ks[m], r2s[m] = _fit_k(xi - xi[0], u, max(1e-3, k0 - 2 * dk), k0 + 2 * dk)
    return ks, r2s, r2s >= min_r2


# --------------------------------------------------------------------------
# Theory (comparison overlays only; the measurement does not use these)
# --------------------------------------------------------------------------

def theory_transverse(k, sp):
    """Stiff string: w^2 = (T/mu) k^2 + (EI/mu) k^4, with the tension and
    bending stiffness the shader actually applies at rest (T_eff, EI_eff,
    see post_process.make_string_params). Returns w in rad/s."""
    T, EI = sp.get("T_eff", sp["T"]), sp.get("EI_eff", sp["EI"])
    return np.sqrt(T / sp["mu"] * k ** 2 + EI / sp["mu"] * k ** 4)


def theory_longitudinal(k, sp):
    """Longitudinal waves: w = sqrt(K/mu) k with K the tangent stiffness of the
    shader's axial force at rest (EA_eff). Returns w in rad/s."""
    return np.sqrt(sp.get("EA_eff", sp["EA"]) / sp["mu"]) * k


def driven_mask(f_peaks, f_sources, tol):
    """Peaks within tol of a source frequency (measured data only)."""
    f_peaks = np.asarray(f_peaks, np.float64)
    f_sources = np.asarray(f_sources, np.float64)
    if f_peaks.size == 0 or f_sources.size == 0:
        return np.zeros(f_peaks.shape, bool)
    return np.min(np.abs(f_peaks[:, None] - f_sources[None, :]), axis=1) <= tol


# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------

def _analyse(dispersion, which):
    freqs = dispersion["freqs"]
    rows, f_res = find_resonances(freqs, dispersion[f"psd_{which}"])
    k_map = k_from_kw_rows(dispersion[f"kw_{which}"], dispersion["k_grid"], rows)
    # Below ~1.5x the k resolution of the interior window (2pi / L_interior,
    # interior = 90% of the string with kspace_grid's default margin) the map
    # peak merges with k=0 and its mirror; those estimates are dropped.
    k_res = 2 * np.pi / (0.9 * dispersion["length"])
    k_map[k_map < 1.5 * k_res] = np.nan
    k_fit, r2, ok = k_from_mode_shapes(dispersion[f"cross_{which}"], rows,
                                       dispersion["node_count"], dispersion["length"])
    return 2 * np.pi * f_res, k_map, k_fit, ok


def plot_dispersion(dispersion, string_params, normalized=False, f1=None,
                    title_prefix="", filename=""):
    """k-w map (top) and phase velocity w/k (bottom), transverse and
    longitudinal side by side. Hollow markers: k from the k-w map; filled:
    k from mode-shape fits. Reference curves use the stiffnesses the shader
    applies at rest; they are comparison overlays only.

    normalized: x = mode number n = kL/pi, map y = w / w1 (w1 = measured
    fundamental, f1), velocity = v / c0 with c0 = sqrt(T/mu) resp. sqrt(EA/mu);
    the transverse velocity curve is then sqrt(1 + B n^2), the inharmonicity.
    Peaks driven by the other polarisation are shown grey on the map and left
    out of the velocity panel.

    "Driven" is decided from measured data only: a longitudinal peak at a
    combination frequency (f_m + f_n, |f_m - f_n|, 2 f_n) of the measured
    transverse resonances (the coupling force is quadratic in the transverse
    motion), a transverse peak at a measured free longitudinal resonance;
    within one Welch frequency bin.
    """
    freqs = dispersion["freqs"]
    k_grid = dispersion["k_grid"]
    w_axis = 2 * np.pi * freqs
    has_theory = string_params.get("T") is not None
    L = string_params["L"]
    if f1 is None:
        _, f_res = find_resonances(freqs, dispersion["psd_trans"])
        f1 = float(f_res[0]) if f_res.size else freqs[1]
    w1 = 2 * np.pi * f1

    # Model-free driven-peak classification (see docstring).
    from plots.energy_flow import combination_frequencies
    tol = freqs[1] - freqs[0]
    f_t = find_resonances(freqs, dispersion["psd_trans"])[1]
    f_l = find_resonances(freqs, dispersion["psd_long"])[1]
    long_driven = driven_mask(f_l, combination_frequencies(f_t), tol)
    trans_driven = driven_mask(f_t, f_l[~long_driven], tol)

    fig, axes = plt.subplots(2, 2, figsize=(17, 11), sharex='col', layout='constrained',
                             gridspec_kw={"height_ratios": [3, 2]})
    fig.get_layout_engine().set(wspace=0.12)
    for col, (which, label, theory_fn, c0_key, c0_label) in enumerate((
            ("trans", "Transverse (Y+Z)", theory_transverse, "T_eff", "√(T/μ)"),
            ("long", "Longitudinal (X)", theory_longitudinal, "EA_eff", "√(K/μ)"))):
        w_res, k_map, k_fit, ok = _analyse(dispersion, which)
        kw = dispersion[f"kw_{which}"].astype(np.float64)
        c0_val = string_params.get(c0_key, string_params.get(c0_key[:-4]))
        c0 = np.sqrt(c0_val / string_params["mu"]) if has_theory and c0_val else None

        # Axis transforms (identity in physical units).
        fx = (lambda k: k * L / np.pi) if normalized else (lambda k: k)
        fy = (lambda w: w / w1) if normalized else (lambda w: w)
        fv = (lambda v: v / c0) if (normalized and c0) else (lambda v: v)

        driven = long_driven if which == "long" else trans_driven
        other = "transverse" if which == "long" else "longitudinal"
        free = ok & ~driven
        k_map_free = np.where(driven, np.nan, k_map)

        # Data-driven ranges: a little beyond the highest measured resonance.
        k_meas = np.concatenate([k_map_free[np.isfinite(k_map_free)], k_fit[free]])
        w_max = min(w_axis[-1], 1.1 * w_res.max()) if w_res.size else w_axis[-1]
        k_lim = min(k_grid[-1], 1.1 * k_meas.max()) if k_meas.size else k_grid[-1]

        # k-w map, each frequency row normalised to its own peak (the ridge
        # stays visible as amplitudes fall with frequency); rows more than
        # 60 dB below the strongest are drawn at the floor, not amplified.
        wmask = w_axis <= w_max
        p = kw[wmask]
        row_max = p.max(axis=1, keepdims=True)
        dead = row_max[:, 0] < 1e-6 * kw.max() if kw.max() > 0 else np.ones(p.shape[0], bool)
        row_max[row_max <= 0] = 1.0
        db = 10.0 * np.log10(np.maximum(p / row_max, 1e-6))
        db[dead] = -60.0
        ax = axes[0, col]
        pcm = ax.pcolormesh(fx(k_grid), fy(w_axis[wmask]), db, cmap='magma', vmin=-60, vmax=0,
                            shading='nearest')
        # Colourbar right of the f (Hz) axis, outside the layout so the
        # velocity panel underneath keeps the same width.
        cax = ax.inset_axes([1.17, 0.0, 0.025, 1.0])
        fig.colorbar(pcm, cax=cax, label="|U(k,ω)|² (dB re. row peak)")

        k_th = np.linspace(k_grid[0], k_lim, 400)[1:]
        theory = theory_fn(k_th, string_params) if has_theory else None
        ax.scatter(fx(k_map_free), fy(w_res), s=22, facecolors='none', edgecolors='cyan', linewidths=0.8,
                   label="k from k–ω map", zorder=3)
        ax.scatter(fx(k_fit[free]), fy(w_res[free]), s=7, color='lime', label="k from mode-shape fit", zorder=4)
        if driven.any():
            k_drv = np.where(np.isfinite(k_fit[driven]), k_fit[driven], k_map[driven])
            ax.scatter(fx(k_drv), fy(w_res[driven]), s=40, facecolors='none', edgecolors='#8f8e88',
                       linewidths=1.2, label=f"driven by {other} motion", zorder=4)
        if theory is not None:
            ax.plot(fx(k_th), fy(theory), color='white', linestyle='--', linewidth=1.0,
                    label="reference (shader parameters)", zorder=5)
        ax.set_ylim(0, fy(w_max))
        if normalized:
            ax.set_ylabel("ω / ω₁")
            sec = ax.secondary_yaxis('right', functions=(lambda v: v * f1, lambda v: v / f1))
            top = ax.secondary_xaxis('top', functions=(lambda n: n * np.pi / L, lambda k: k * L / np.pi))
            top.set_xlabel("Wavenumber k (rad/m)")
        else:
            ax.set_ylabel("ω (rad/s)")
            sec = ax.secondary_yaxis('right', functions=(lambda v: v / (2 * np.pi), lambda v: v * 2 * np.pi))
        sec.set_ylabel("f (Hz)")
        ax.set_title(f"{label}: {w_res.size} resonances, {int(ok.sum())} shape fits accepted, "
                     f"{int(driven.sum())} driven", pad=26)
        ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=4, frameon=False, fontsize=8)

        # Phase velocity (free waves only).
        ax = axes[1, col]
        with np.errstate(divide='ignore', invalid='ignore'):
            v_map = w_res / k_map_free
            v_fit = w_res[free] / k_fit[free]
        ax.scatter(fx(k_map_free), fv(v_map), s=22, facecolors='none', edgecolors='tab:blue', linewidths=0.8,
                   label="k–ω map")
        ax.scatter(fx(k_fit[free]), fv(v_fit), s=9, color='tab:green', label="Mode-shape fit", zorder=3)
        vals = [fv(v_fit)]
        if theory is not None:
            ax.plot(fx(k_th), fv(theory / k_th), color='black', linestyle='--', linewidth=1.0,
                    label="reference (shader parameters)")
            ax.axhline(fv(c0), color='tab:gray', linestyle=':', linewidth=1.0, label=c0_label)
            vals += [fv(theory / k_th), [fv(c0)]]
        vals = np.concatenate([np.ravel(v) for v in vals])
        vals = vals[np.isfinite(vals)]
        if vals.size:
            lo, hi = np.min(vals), np.max(vals)
            pad = max(0.05 * (hi - lo), 0.01 * hi)
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(0, fx(k_lim))
        if normalized:
            ax.set_xlabel("Mode number n = kL/π")
            ax.set_ylabel("Phase velocity v / c₀")
            if c0:
                sv = ax.secondary_yaxis('right', functions=(lambda v, c=c0: v * c, lambda v, c=c0: v / c))
                sv.set_ylabel("m/s")
        else:
            ax.set_xlabel("Wavenumber k (rad/m)")
            ax.set_ylabel("Phase velocity ω/k (m/s)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=4, frameon=False, fontsize=8)

    kind = "Normalised Dispersion Relation" if normalized else "Dispersion Relation"
    fig.suptitle(f"[{title_prefix}] {kind}")
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return [filename]
