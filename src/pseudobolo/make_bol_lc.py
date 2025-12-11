from __future__ import annotations

import numpy as np
from dataclasses import dataclass
import pathlib
from typing import List, Optional, Sequence, Tuple
from scipy.interpolate import interp1d, UnivariateSpline
from pseudobolo.aux import al_av, estimate_56ni, filter_shortname, create_lc_df
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.widgets import Button
import pandas as pd



#--------------------------------------------------
# Attempt at translating Stéphane's IDL code into Python, started 19.11.25, current version 11.12.25
#----------------------------------------------------------------


# -------------------------------------------------------------------
# Core function: Python version of mklcbol.pro
# -------------------------------------------------------------------
EPS_TIME = 1e-2  # IDL: 1d-2


def _interp_mag_linear_with_error(
    time: np.ndarray,
    mag: np.ndarray,
    magerr: np.ndarray,
    time_interp: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simplified version of the IDL 'l' branch:
    - if |t - measurement time| < EPS_TIME -> use that point directly
    - else: bracket and do *simple* linear interpolation for value
            and linear interpolation of errors (approximation)
    """
    nt = len(time_interp)
    y_out = np.empty(nt, dtype=float)
    yerr_out = np.empty(nt, dtype=float)

    for ii, t in enumerate(time_interp):
        # exact (or nearly exact) data point?
        diff = np.abs(time - t)
        j = np.argmin(diff)
        if diff[j] < EPS_TIME:
            y_out[ii] = mag[j]
            yerr_out[ii] = magerr[j]
            continue

        # bracket
        # indices with time <= t and >= t
        left_candidates = np.where(time <= t)[0]
        right_candidates = np.where(time >= t)[0]
        if len(left_candidates) == 0 or len(right_candidates) == 0:
            # out-of-range -> extrapolate using np.interp (same as nearest bracket)
            y_out[ii] = np.interp(t, time, mag)
            yerr_out[ii] = np.interp(t, time, magerr)
            continue

        rri = left_candidates.max()
        rrs = right_candidates.min()
        x1, x2 = time[rri], time[rrs]
        y1, y2 = mag[rri], mag[rrs]
        e1, e2 = magerr[rri], magerr[rrs]

        if x2 == x1:
            y_out[ii] = y1
            yerr_out[ii] = e1
        else:
            w = (t - x1) / (x2 - x1)
            y_out[ii] = (1 - w) * y1 + w * y2
            # very simple error propagation (linear interp of errors)
            yerr_out[ii] = np.sqrt((1 - w) ** 2 * e1 ** 2 + w ** 2 * e2 ** 2)

    return y_out, yerr_out

def _interp_mag_gp(
    time: np.ndarray,
    mag: np.ndarray,
    magerr: np.ndarray,
    time_interp: np.ndarray,
    explosion_time: Optional[float] = None,
    length_scale: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate magnitudes with a Gaussian process.

    Parameters
    ----------
    time : array
        Observation times (MJD).
    mag : array
        Magnitudes at those times.
    magerr : array
        1σ errors on the magnitudes.
    time_interp : array
        Times at which to predict.
    length_scale : float
        Typical correlation timescale (days) of the light curve.
        Smaller => more wiggles, larger => smoother.

    Returns
    -------
    mu : array
        GP mean magnitudes at time_interp.
    sigma : array
        GP 1σ uncertainty at time_interp.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF
    # --- Sort and convert to arrays ---
    order = np.argsort(time)
    t = np.asarray(time[order], dtype=float)
    y = np.asarray(mag[order], dtype=float)
    e = np.asarray(magerr[order], dtype=float)
    t_star = np.asarray(time_interp, dtype=float)

    # --- Center + scale time for numerical stability ---
    #   keep all points, just rescale units internally
    t0 = t.mean()
    t_scaled = (t - t0) / length_scale        # dimensionless
    t_star_scaled = (t_star - t0) / length_scale

    X = t_scaled[:, None]        # (N, 1)  <-- fixes the 1D/2D error
    X_star = t_star_scaled[:, None]

    # --- Kernel: RBF with fixed amplitude and length scale = 1 in scaled units ---
    kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(
        length_scale=1.0,
        length_scale_bounds="fixed",
    )

    # --- Noise: measurement variance + small jitter ---
    # avoid zeros/NaNs in magerr
    e_safe = e.copy()
    bad = ~np.isfinite(e_safe) | (e_safe <= 0)
    if np.any(bad):
        # fallback to median of good errors
        med = np.nanmedian(e_safe[~bad]) if np.any(~bad) else 0.1
        e_safe[bad] = med
    alpha = e_safe**2 + (0.01)**2  # tiny jitter for numerical stability

    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        normalize_y=True,
        optimizer=None,     # do NOT refit hyperparameters
    )

    # --- Fit & predict on the full common time grid ---
    gpr.fit(X, y)
    mu, sigma = gpr.predict(X_star, return_std=True)
    return mu, sigma 

def _interp_mag_any(
    time: np.ndarray,
    mag: np.ndarray,
    magerr: np.array,
    time_interp: np.ndarray,
    interpmeth: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    SciPy-based interpolation for interpmeth in {'c', 'u', 's', ...}.

    Parameters
    ----------
    time : array
        Observation times.
    mag : array
        Magnitudes at those times.
    magerr : array
        1σ uncertainties on mag (same length as time).
    time_interp : array
        Times at which to interpolate.
    interpmeth : {'l', 'g', 'c','u','s',...}
        'l': linear
        'g': Gaussian process
        'c' : global least-squares quadratic (like /lsquadratic)
        'u' : piecewise quadratic (interp1d(kind='quadratic'))
        's' : cubic spline (UnivariateSpline)
        else : linear interpolation fallback

    Returns
    -------
    y : array
        Interpolated magnitudes at time_interp.
    yerr : array
        Interpolated magnitude errors at time_interp (approximate for 'u'/'s').
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF

    # Ensure sorted, unique x for interpolation
    idx = np.argsort(time)
    t_sorted = time[idx]
    y_sorted = mag[idx]
    yerr_sorted = magerr[idx]

    # Drop exact duplicates in t if they exist (optional but safer for splines)
    # (keep last occurrence)
    _, unique_idx = np.unique(t_sorted, return_index=True)
    t = t_sorted[unique_idx]
    y = y_sorted[unique_idx]
    e = yerr_sorted[unique_idx]

    m = interpmeth.lower()
    
    if m =="c": m =="u"
    
    # --- 'c' : global least-squares quadratic (lsquadratic in IDL) -------------
    '''if m == "c": #  TO DO: FIGURE OUT WHY THIS IS NOT WORKING ----
        # Shift time axis to improve conditioning of the quadratic fit
        t0 = np.mean(t)
        tt = t - t0                  # centred times for fitting
        x = time_interp - t0         # centred times for evaluation

        # Avoid zero/NaN weights
        w = np.where(e > 0, 1.0 / e, 0.0)

        # Fit y = a0 + a1 * tt + a2 * tt^2
        coeffs, cov = np.polyfit(tt, y, deg=2, w=w, cov=True)
        a0, a1, a2 = coeffs

        # Interpolated values at x = (time_interp - t0)
        y_out = a0 + a1 * x + a2 * x**2

        # Error propagation: var(y) = v^T C v, v = [1, x, x^2]
        v = np.vstack([np.ones_like(x), x, x**2])  # (3, N)
        Cv = cov @ v                               # (3,3) @ (3,N) -> (3,N)
        var = np.sum(v * Cv, axis=0)               # (N,)
        var = np.clip(var, 0.0, np.inf)
        yerr_out = np.sqrt(var)

        return y_out, yerr_out'''
    # -------- 'l': linear -----------------------------------------------------
    if m == "l":
        y_out, yerr_out = _interp_mag_linear_with_error(t, y, e, time_interp)
        return y_out, yerr_out
    
    # ----'g': Gaussian Process ------------------------------------------------
    if m =="g":
        y_out, yerr_out = _interp_mag_gp(t, y, e, time_interp)
        return y_out, yerr_out
    # --- 'u' : quadratic (piecewise) -------------------------------------------

    elif m == "u":
        # /quadratic: piecewise quadratic interpolation
        # interp1d(kind='quadratic') uses quadratic splines segment-wise.
        # Quadratic interpolation of the magnitudes
        f_mag = interp1d(
            t, y,
            kind="quadratic",
            bounds_error=False,
            fill_value="extrapolate",
        )
        y_out = f_mag(time_interp)

        # Approximate error interpolation using the same scheme
        # (this is not a strict propagation, but close in spirit to IDL's "ignore"
        # and still gives you some time structure in the uncertainties)
        f_err = interp1d(
            t, e,
            kind="quadratic",
            bounds_error=False,
            fill_value="extrapolate",
        )
        yerr_out = np.abs(f_err(time_interp))

        return y_out, yerr_out

    # --- 's' : cubic spline ----------------------------------------------------
    elif m == "s":
        # Spline through the points; you can optionally use weights 1/e
        # but set s=0 to pass exactly through the data (like an interpolating spline)
        # If some errors are zero, avoid infinite weights
        w = np.where(e > 0, 1.0 / e, 1.0)
        spline = UnivariateSpline(t, y, w=w, k=3, s=0)
        y_out = spline(time_interp)

        # For the errors, use a separate spline on e (or just linear interp if you prefer)
        spline_err = UnivariateSpline(t, e, k=3, s=0)
        yerr_out = np.abs(spline_err(time_interp))

        return y_out, yerr_out

    # --- default: linear interpolation ----------------------------------------
    else:
        f_mag = interp1d(
            t, y,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        y_out = f_mag(time_interp)

        f_err = interp1d(
            t, e,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        yerr_out = np.abs(f_err(time_interp))

        return y_out, yerr_out

class PseudoBoloWizard:
    def __init__(
        self,
        lightcurve_file: str | pathlib.Path,
        pbinfo_file: str | pathlib.Path | None = None,
        debug: bool = False,
    ):
        """
        GUI wizard for building a pseudo-bolometric light curve.

        Parameters
        ----------
        lightcurve_file : str or Path
            Path to lcbolinput-style .dat file (header + per-filter blocks).
        pbinfo_file : str or Path or None
            Path to pbinfo.dat.
            If None, a default tests/tests_data/pbinfo.dat is used.
        debug : bool
            Extra prints, etc.
        """
        self.debug = debug
        self.infile = pathlib.Path(lightcurve_file)
        if pbinfo_file is None:
            here = pathlib.Path(__file__).resolve()
            repo_root = here.parents[4]  # adjust if needed
            self.pbinfo_path = repo_root / "tests" / "tests_data" / "pbinfo.dat"
        else:
            self.pbinfo_path = pathlib.Path(pbinfo_file).expanduser().resolve()
        

        # --- Build dataframes + header from files ---     
        self.hdr, self.lightcurve_df, self.pbinfo = create_lc_df(self.infile, self.pbinfo_path)
                
        self.filters = sorted(self.lightcurve_df["filter"].unique())

        # storage for GP results, step tracking, etc.
        self.step = 1
        self._step_widgets: list = []
        self.selected: dict[str, bool] = {f: True for f in self.filters}
        self.gp_interp = {} #saves the per filter GP interpolations
        self.mjd_min = self.lightcurve_df["mjd"].min()
        self.mjd_max = self.lightcurve_df["mjd"].max()
        self.time_grid = None  # will be set from _create_time_grid
        self.coverage_patches = {}

        # --- Build figure + first step ---
        self._build_figure()
        self._build_step1()  # filter selection
        
    @property
    def selected_filters(self) -> list[str]:
        """Canonical list of currently selected filters."""
        return [f for f, sel in self.selected.items() if sel]
    
    

    def _build_figure(self):
        self.fig = plt.figure(figsize=(12, 9))
        self.fig.canvas.manager.set_window_title("Bolometric LC wizard")

        # More room at bottom for TWO rows of buttons
        self.fig.subplots_adjust(
            left=0.06,
            right=0.97,
            top=0.88,
            bottom=0.18,   # <- keeps plots away from buttons
            hspace=1.0,   # <- bigger vertical space between rows of subplots
            wspace=0.30,
        )

        #Navigation buttons (Back / Next) – centered below the presets
        self.nav_w   = 0.18
        self.nav_h   = 0.05
        self.pad_nav = 0.04
        self.y_nav   = 0.02

        self.x_back = 0.5 - self.nav_w - self.pad_nav/2
        self.x_next = 0.5 + self.pad_nav/2

        ax_back = self.fig.add_axes([self.x_back, self.y_nav, self.nav_w, self.nav_h])
        ax_next = self.fig.add_axes([self.x_next, self.y_nav, self.nav_w, self.nav_h])

        self.btn_back = Button(ax_back, "Back")
        self.btn_back.label.set_fontweight("bold")

        self.btn_next = Button(ax_next, "Next")
        self.btn_next.label.set_fontweight("bold")

        self.btn_back.on_clicked(self._on_back)
        self.btn_next.on_clicked(self._on_next)



    def _clear_axes(self):
        # Remove all existing axes except the button axis
        # only clear the *plot* axes, not the whole figure
            
        # Axes we always keep: Back / Next buttons
        keep = {self.btn_back.ax, self.btn_next.ax}
        
        # Remove everything else
        for ax in list(self.fig.axes):
            if ax not in keep:
                ax.remove()

        # Reset references
        # Reset bookkeeping for data axes
        self.axes_step1 = []
        self.axes_step2 = []
        self.axes_step3 = []
        self.coverage_ax = None
        self.coverage_patches.clear()
        
    def _clear_step_widgets(self):
        """Remove step-specific widgets (preset buttons, sliders, etc)."""
        for w in getattr(self, "_step_widgets", []):
            try:
                w.disconnect_events()
            except Exception:
                pass
            try:
                w.ax.remove()
            except Exception:
                pass
        self.step_widgets = []
            
    def _get_pbinfo(self, filter_name):
        """Retrieve lambda_eff and ew for a given filter_name from pbinfo."""
        # Case 1: pbinfo is a DataFrame with filter_name as index
        pbinfo = self.pbinfo
        if hasattr(pbinfo, "loc"):
            row = pbinfo.loc[filter_name]
            lam_eff = row["lambda_eff"]
            ew = row["ew"]
        else:
            # Case 2: pbinfo is a dict-like mapping filter_name -> dict/obj
            info = pbinfo[filter_name]
            lam_eff = info["lambda_eff"]
            ew = info["ew"]
        return lam_eff, ew
    
    def _apply_selection_visuals_step1(self):
        for ax, f in self.ax_to_filter.items():
            ax.set_facecolor("white" if self.selected[f] else "#dddddd")

        for f, patch in self.coverage_patches.items():
            patch.set_alpha(0.8 if self.selected.get(f, False) else 0.2)
        self._update_interp_region_step1()
        self.fig.canvas.draw_idle()
        
    def _update_interp_region_step1(self):
        # compute time grid based on currently selected filters
        self.time_grid = self._create_time_grid()
        # clean up old patches if t_grid is empty or too short
        if self.time_grid.size == 0:
            # remove any existing patches
            if hasattr(self, "interp_region_patches"):
                for p in self.interp_region_patches.values():
                    p.remove()
                self.interp_region_patches.clear()
            return

        t_min, t_max = self.time_grid.min(), self.time_grid.max()

        if not hasattr(self, "interp_region_patches"):
            self.interp_region_patches = {}

        for ax in self.axes_step1:
            # remove old patch if present
            old = self.interp_region_patches.pop(ax, None)
            if old is not None:
                old.remove()
            # add new patch
            patch = ax.axvspan(
                t_min, t_max,
                alpha=0.12,
                color = '#9EC5AB',
                zorder=0  # behind data
            )
            self.interp_region_patches[ax] = patch
        
    # Helper: resolve patterns to actual filter names
    def _filters_for_patterns(self, patterns):
        """
        patterns is a list like ['U','B','V','R','I'] or ['*'].
        Matching rule:
        '*'  -> all filters
        'K'  -> any filter starting with 'K' (e.g. 'Ks_2MASS')
        """
        if "*" in patterns:
            return list(self.filters)
        chosen = []
        for f in self.filters:
            for p in patterns:
                if f.startswith(p):
                    chosen.append(f)
                    break
        return sorted(set(chosen))
    
    def _make_preset_callback(self, label):
        pats = self.preset_groups.get(label, [])

        def _callback(event):
            # Use your helper that maps patterns → full filter names
            chosen = self._filters_for_patterns(pats)
            # Turn ON chosen, OFF all others
            for f in self.filters:
                self.selected[f] = (f in chosen)
            self._apply_selection_visuals_step1()

        return _callback
    
    def _create_time_grid(self, filters=None):
        """
        Build a global time grid from the given filters.
        If filters is None, use currently selected filters (or all filters as fallback).
        """
        # Decide which filters to use
        if filters is None:
            if hasattr(self, "selected") and self.selected:
                filters = [f for f, ok in self.selected.items() if ok]
            else:
                filters = getattr(self, "filters", [])

        if not filters:
            return np.array([])

        sub_all = self.lightcurve_df[self.lightcurve_df["filter"].isin(filters)]
        if sub_all.empty:
            return np.array([])

        t_grid = np.sort(sub_all["mjd"].unique())

        # restrict to region where *all* filters still have data
        t_max = sub_all.groupby("filter")["mjd"].max().min()
        t_grid = t_grid[t_grid <= t_max]
        return t_grid
    
    
    
    def _on_next(self, event):
        print(f"\nNext button clicked at step {self.step}.")
        if self.fig is None:
            return

        if self.step == 1:
            filters = self.selected_filters
            print("Selected filters:", filters)
            if not filters:
                print("No filters selected, ignoring Next.")
                return

            # common time grid from step 1 selection
            self.time_grid = self._create_time_grid(filters)

            self.step = 2
            print(f"Moving to step {self.step}.")
            self._build_step2()

        elif self.step == 2:
            self.step = 3
            print(f"Moving to step {self.step}.")
            self._build_step3()

        elif self.step == 3:
            self.step = 4
            print(f"Moving to step {self.step}.")
            self.btn_next.label.set_text("Exit")
            self._build_step4()

        else:
            print("Finishing wizard.")
            import matplotlib.pyplot as plt
            plt.close(self.fig)
        
        
    def _on_back(self, event):
        print(f"Back button clicked at step {self.step}.")
        if self.fig is None:
            return

        if self.step == 2:
            self.step = 1
            print("Returning to step 1.")
            self.btn_next.label.set_text("Next")
            self._build_step1()

        elif self.step == 3:
            self.step = 2
            print("Returning to step 2.")
            self.btn_next.label.set_text("Next")  # or "Next" vs "Finish" logic later
            self._build_step2()
            
        elif self.step == 4:
            self.step = 3
            print("Returning to step 3.")
            self.btn_next.label.set_text("Next")
            self._build_step3()


    # ---------- STEP 1: filter selection ----------
    def _build_step1(self):
        # make sure any old step widgets are gone
        self._clear_step_widgets()
        self._clear_axes()
        self.fig.suptitle("Step 1: Select filters\n"
                          "Click a panel to toggle selection (grey = OFF).\n"
                          "Press 'Next' when done.",
                          y=0.99,)

    
        # --- Determine filters and layout ---
        self.ncols = 3
        filters_unique = sorted(self.lightcurve_df["filter"].unique())
        # Build (filter, lambda_eff) list
        f_lam = []
        for f in filters_unique:
            lam_eff, ew = self._get_pbinfo(f) 
            f_lam.append((f, lam_eff))

        # Sort by lambda_eff
        f_lam_sorted = sorted(f_lam, key=lambda t: t[1])

        # Keep only the filter names in the new order
        filters = [f for (f, _) in f_lam_sorted]
        self.filters = filters
        n_filt = len(filters)
        
        # Total rows: 1 (coverage) + nrows_filters (per-filter plots)
        self.ncols = min(self.ncols, n_filt)
        self.nrows_filters = int(np.ceil(n_filt / self.ncols))
        
        if n_filt == 0:
            raise ValueError("No filters found in lightcurve_df['filter'].")
        
        self.preset_groups = {
                "UBVRI": ["U", "B", "V", "R", "I"],
                "BVRI":  ["B", "V", "R", "I"],
                "JHK":   ["J", "H", "K"],   # we’ll also match 'Ks'
                "All":   ["*"],        # '*' = all filters
            }


        self.gs = self.fig.add_gridspec(
            self.nrows_filters + 1,
            self.ncols,
            top=0.90,
            bottom=0.21,          # <- keep clear of preset + nav buttons
            left=0.08,
            right=0.98,
            height_ratios=[1.4] + [1.0] * self.nrows_filters,
            hspace=0.70,          # <- extra space between coverage & panels
        )
        

        # --- Top coverage axis spanning all columns ---
        self.coverage_ax = self.fig.add_subplot(self.gs[0, :])
        self.coverage_ax.set_xlabel("Wavelength [$\\AA$]")
        self.coverage_ax.set_ylabel("Transmission")
        self.coverage_ax.grid(True, alpha = 0.1)
        self.coverage_ax.set_title("Filter Transmission Blocks")
        self.coverage_ax.set_ylim(0, 1.2)

        # Track selection state and artists
        self.selected = {f: True for f in filters}
        self.coverage_patches = {}
        self.axes_step1 = []
        self.ax_to_filter = {}        

        # --- Plot transmission curves for each filter ---
        for f in filters:
            lam_eff, ew = self._get_pbinfo(f)
            short = filter_shortname(f)
            lam_min = lam_eff - (ew/2.0)
            lam_max = lam_eff + (ew/2.0)
            wave = np.linspace(lam_min, lam_max, 50)
            poly = self.coverage_ax.fill_between(wave, 0, 1, alpha = 0.8)
            self.coverage_ax.annotate(f"{short}", xy=(lam_eff, 1.05), xycoords="data", ha = "center", fontsize="small")
            self.coverage_patches[f] = poly

        # --- Per-filter subplots below ---

        idx = 0
        
        for r in range(self.nrows_filters):
            for c in range(self.ncols):
                if idx >= n_filt:
                    break

                f = filters[idx]
                ax = self.fig.add_subplot(self.gs[r + 1, c])
                ax.set_xlabel("MJD")
                ax.set_ylabel("Magnitude")
                self.axes_step1.append(ax)
                self.ax_to_filter[ax] = f

                sub = self.lightcurve_df[self.lightcurve_df["filter"] == f]

                if "err" in sub.columns:
                    ax.errorbar(sub["mjd"], sub["mag"], yerr=sub["err"], fmt="o", ms=3)
                else:
                    ax.plot(sub["mjd"], sub["mag"], "o", ms=3)

                ax.set_title(f)
                ax.invert_yaxis()
                ax.grid(True, alpha=0.1)

                idx += 1
                
        for ax in self.axes_step1:
            ax.set_xlim(self.mjd_min - 2, self.mjd_max + 2)

        # Hide any unused grid cells
        total_axes = (self.nrows_filters) * self.ncols
        for extra in range(idx, total_axes):
            r = extra // self.ncols
            c = extra % self.ncols
            ax = self.fig.add_subplot(self.gs[r + 1, c])
            ax.set_visible(False)


        self._apply_selection_visuals_step1()
        self._update_interp_region_step1() 
        self.cid_click_step1 = self.fig.canvas.mpl_connect(
        "button_press_event", self._on_click_step1
        )

        # ------------------------------------------------------------------
        # Buttons
        # ------------------------------------------------------------------

        # 1) Preset buttons (UBVRI/BVRI/JHK/All) – centered row
        btn_w  = 0.13
        btn_h  = 0.05
        pad_x  = 0.02          # horizontal spacing between preset buttons
        y_presets = 0.1       # vertical position of the preset row (above Next/Back)

        total_w = 4 * btn_w + 3 * pad_x
        x0 = 0.5 - total_w / 2  # left edge so that the row is centered around x=0.5

        ax_ubvri = self.fig.add_axes([x0 + 0*(btn_w+pad_x), y_presets, btn_w, btn_h])
        ax_bvri  = self.fig.add_axes([x0 + 1*(btn_w+pad_x), y_presets, btn_w, btn_h])
        ax_jhk   = self.fig.add_axes([x0 + 2*(btn_w+pad_x), y_presets, btn_w, btn_h])
        ax_all   = self.fig.add_axes([x0 + 3*(btn_w+pad_x), y_presets, btn_w, btn_h])

        self.btn_ubvri = Button(ax_ubvri, "UBVRI")
        self.btn_bvri  = Button(ax_bvri,  "BVRI")
        self.btn_jhk   = Button(ax_jhk,   "JHK")
        self.btn_all   = Button(ax_all,   "All")

        self.btn_ubvri.on_clicked(self._make_preset_callback("UBVRI"))
        self.btn_bvri.on_clicked(self._make_preset_callback("BVRI"))
        self.btn_jhk.on_clicked(self._make_preset_callback("JHK"))
        self.btn_all.on_clicked(self._make_preset_callback("All"))
        
        self._step_widgets = [
            self.btn_ubvri,
            self.btn_bvri,
            self.btn_jhk,
            self.btn_all,
        ]
        
        self.cid_click_step1 = self.fig.canvas.mpl_connect(
            "button_press_event", self._on_click_step1)
        self.fig.canvas.draw_idle()



    def _on_click_step1(self, event):
        
        ax = event.inaxes
        if ax is None:
            return

        # 1) Click in a per-filter panel
        if ax in self.ax_to_filter:
            f = self.ax_to_filter[ax]
            self.selected[f] = not self.selected[f]
            self._apply_selection_visuals_step1()
            return

        # 2) Click in the coverage axis: toggle based on patch
        if ax is self.coverage_ax:
            for f, patch in self.coverage_patches.items():
                contains, _ = patch.contains(event)
                if contains:
                    self.selected[f] = not self.selected[f]
                    self._apply_selection_visuals_step1()
                    break

    # ---------- STEP 2: show original + interpolated data for each selected filter. ----------
    def _build_step2(self):
        """Step 2: show original + GP-interpolated data for each selected filter,
        using a *common* time grid consisting of all measurement epochs.
        """
        self._clear_step_widgets()
        self._clear_axes()
        # Stop listening for step-1 clicks
        if hasattr(self, "cid_click_step1"):
            self.fig.canvas.mpl_disconnect(self.cid_click_step1)
        
        print("Building step 2: GP interpolation inspection.")

        self.fig.suptitle(
            "Step 2: Inspect interpolation\n"
            "Markers = original data, line = GP mean, band = $1\\sigma$\n"
            "Press 'Finish' when done.",
            y=0.99,
        )

        # Filters to show = those selected in step 1
        filters = self.selected_filters or self.filters
        n_filt = len(filters)
        if n_filt == 0:
            raise ValueError("No filters to show in step 2.")


        # Layout: same ncols, as many rows as needed
        self.ncols = min(self.ncols, n_filt)
        nrows = int(np.ceil(n_filt / self.ncols))

        # New GridSpec for step 2 (no coverage row)
        self.gs2 = self.fig.add_gridspec(
            nrows, self.ncols,
            height_ratios=[1.0] * nrows,
        )

        self.axes_step2 = []

        idx = 0
        for r in range(nrows):
            for c in range(self.ncols):
                if idx >= n_filt:
                    break

                f = filters[idx]
                ax = self.fig.add_subplot(self.gs2[r, c])
                self.axes_step2.append(ax)

                # --- original data for this filter ---
                sub = self.lightcurve_df[self.lightcurve_df["filter"] == f]

                if sub.empty:
                    ax.set_title(filter_shortname(f) + " (no data)")
                    ax.grid(True)
                    idx += 1
                    continue

                t_data = sub["mjd"].values
                m_data = sub["mag"].values
                e_data = sub["err"].values if "err" in sub.columns else None

                if e_data is not None:
                    ax.errorbar(
                        t_data, m_data, yerr=e_data,
                        fmt="o", ms=3, label="Original"
                    )
                else:
                    ax.plot(
                        t_data, m_data,
                        "o", ms=3, label="Original"
                    )

                # --- GP interpolation on the *common* time grid ---
                try:
                    if e_data is None:
                        # if no errors, assume some small error
                        e_data_use = np.full_like(m_data, 0.05, dtype=float)
                    else:
                        e_data_use = e_data
                    
                    mu, sigma = _interp_mag_gp(
                        time=t_data,
                        mag=m_data,
                        magerr=e_data_use,
                        time_interp=self.time_grid,
                        explosion_time=None,   # or a per-filter value if you have it
                        length_scale=10.0,
                    )
                    self.gp_interp[f] = {"mjd": self.time_grid, "mag": mu, "err": sigma}

                    # Plot GP mean and 1σ band
                    ax.scatter(self.time_grid, mu, s=15, label="GP mean", c = '#BD2D87')  # no "-"
                    ax.fill_between(
                        self.time_grid, mu - sigma, mu + sigma,
                        alpha=0.2, color = '#BD2D87', linewidth=0, label="GP $\\pm 1\\sigma$"
                    )
                    ax.set_ylabel("Magnitude")
                    ax.set_xlabel("MJD")
                    
                except Exception as e:
                    print(f"GP interpolation for filter {f} failed: {e}")

                ax.set_title(filter_shortname(f))
                ax.invert_yaxis()
                ax.grid(True)
                ax.legend(fontsize="x-small")

                idx += 1
        for ax in self.axes_step2:
            ax.set_xlim(self.mjd_min - 2, self.mjd_max + 2)

        # Hide any unused cells
        total_cells = nrows * self.ncols
        for extra in range(idx, total_cells):
            r = extra // self.ncols
            c = extra % self.ncols
            ax = self.fig.add_subplot(self.gs2[r, c])
            ax.set_visible(False)

        self.fig.subplots_adjust(bottom=0.18, top=0.88, hspace=0.4)
        self.fig.canvas.draw_idle()
        
        
    def _compute_fluxes_from_gp(self):
        """
        Convert de-reddened GP magnitudes to flux and extinction error,
        following the original IDL mklcbol logic.
        Populates:
          - self.interpflux      : (nselect, nt)
          - self.interpfluxerr   : (nselect, nt)
          - self.flux_filters    : list of selected filter names
          - self.flux_interp[f]  : per-filter dict with mjd, flux, flux_err
        """
        # filters currently selected in the wizard
        filters = self.selected_filters
        if not filters:
            raise RuntimeError("No filters selected for GP interpolation.")

        # common time grid (you already set this e.g. in step2)
        t_grid = self.time_grid
        nt = len(t_grid)

        nselect = len(filters)
        interpflux = np.zeros((nselect, nt), dtype=float)
        interpfluxerr = np.zeros_like(interpflux)

        hdr = self.hdr

        # prepare a container for per-filter results
        self.flux_interp = {}

        for i, f in enumerate(filters):
            # GP result for this filter from step 2:
            gp = self.gp_interp[f]
            m_interp = np.asarray(gp["mag"], dtype=float)

            if m_interp.shape != (nt,):
                raise ValueError(
                    f"GP magnitudes for filter {f} have shape {m_interp.shape}, "
                    f"expected ({nt},)."
                )

            # passband metadata
            meta = self.pbinfo.loc[f]
            lambda_eff = float(meta["lambda_eff"])
            zpt        = float(meta["zpt"])

            # --- extinction coefficients A_lambda / A_V and their errors ---
            AlAv_host = al_av(lambda_eff, r_v=hdr.rvhost, rverr=hdr.rvhosterr)
            AlAv_MW   = al_av(lambda_eff, r_v=hdr.rvmw,   rverr=hdr.rvmwerr)

            # Assuming al_av returns [value, error]
            AlAv_host_val, AlAv_host_err = AlAv_host[0], AlAv_host[1]
            AlAv_MW_val,   AlAv_MW_err   = AlAv_MW[0],   AlAv_MW[1]

            # A_lambda for host and MW
            Al_host = AlAv_host_val * hdr.avhost
            Al_MW   = AlAv_MW_val   * hdr.avmw
            Al_tot  = Al_host + Al_MW   # scalar

            # --- magnitudes → flux, de-reddened ---
            interpflux[i, :] = 10.0**(-0.4 * (m_interp - Al_tot - zpt))

            # --- propagate extinction error ---
            if hdr.avhost > 0.0:
                Al_host_err = Al_host * np.sqrt(
                    (AlAv_host_err / AlAv_host_val)**2 +
                    (hdr.avhosterr / hdr.avhost)**2
                )
            else:
                Al_host_err = 0.0

            if hdr.avmw > 0.0:
                Al_MW_err = Al_MW * np.sqrt(
                    (AlAv_MW_err / AlAv_MW_val)**2 +
                    (hdr.avmwerr / hdr.avmw)**2
                )
            else:
                Al_MW_err = 0.0

            Al_tot_err = np.sqrt(Al_host_err**2 + Al_MW_err**2)

            if Al_tot != 0:
                # interpfluxerr[i,*] = interpflux[i,*] * (0.4*alog(10)*Al_tot_err/Al_tot)
                fac = 0.4 * np.log(10.0) * (Al_tot_err / Al_tot)
                interpfluxerr[i, :] = interpflux[i, :] * fac
            else:
                # no extinction → no extinction error
                interpfluxerr[i, :] = 0.0

            # store per-filter result as well (nice for plotting step 3)
            self.flux_interp[f] = {
                "mjd": t_grid,
                "flux": interpflux[i, :],
                "flux_err": interpfluxerr[i, :],
            }

        # store the big arrays on the wizard
        self.interpflux = interpflux
        self.interpfluxerr = interpfluxerr
                
    def _integrate_fluxes(self):
        """
        Integrate the fluxes over wavelength, handling gaps and overlaps,
        following the original IDL mklcbol logic.
        Populates:
          - self.flux      : (nt,) bolometric luminosity
          - self.flux_int_err  : (nt,) bolometric luminosity error
        """
        self._compute_fluxes_from_gp()
        if not hasattr(self, "interpflux"):
            raise RuntimeError("GP fluxes not computed yet.")

        filters = self.selected_filters
        if not filters:
            raise RuntimeError("No filters selected when computing integrated flux.")

        # sort by lambda_eff as in IDL:
        filters_sorted = sorted(filters, key=lambda f: self.pbinfo.loc[f, "lambda_eff"])

        idx_map = [filters.index(f) for f in filters_sorted]
        F = self.interpflux[idx_map, :]
        Ferr = self.interpfluxerr[idx_map, :]
        
        nselect, nt = F.shape

        flux_int = np.zeros(nt, dtype=float)
        flux_int_err = np.zeros(nt, dtype=float)

        idxlap = 0          # 1 if current filter overlaps with previous
        wred_prev = None    # store previous filter's red edge

        for i, f in enumerate(filters_sorted):
            meta = self.pbinfo.loc[f]
            lam = float(meta["lambda_eff"])
            ew  = float(meta["ew"])

            # Red edge of this filter
            wred = lam + ew / 2.0

            # Blue edge depends on overlap with previous filter
            if i > 0 and idxlap == 1:
                # continuation from previous overlap
                wblue = wred_prev
            else:
                # normal blue edge
                wblue = lam - ew / 2.0

            if i < nselect - 1:
                # all but last filter: compare with next filter
                f_next = filters_sorted[i + 1]
                meta_next = self.pbinfo.loc[f_next]
                lam_next = float(meta_next["lambda_eff"])
                ew_next  = float(meta_next["ew"])

                wbluenext = lam_next - ew_next / 2.0  # blue edge of next filter

                if wred <= wbluenext:
                    # --- GAP between this filter and the next ---
                    # FILTER segment: [wblue, wred] with this filter's flux
                    width_filt = wred - wblue
                    # GAP segment: [wred, wbluenext] with mean flux
                    width_gap = wbluenext - wred

                    flux_int += width_filt * F[i, :]
                    flux_int += width_gap * 0.5 * (F[i, :] + F[i + 1, :])

                    # Errors:
                    flux_int_err += width_filt * Ferr[i, :]
                    flux_int_err += width_gap * 0.5 * np.sqrt(Ferr[i, :]**2 + Ferr[i + 1, :]**2)

                    idxlap = 0
                else:
                    # --- OVERLAP with next filter ---
                    # Non-overlap part: [wblue, wbluenext] at this filter's flux
                    width_nonoverlap = wbluenext - wblue
                    # Overlap part: [wbluenext, wred] at mean flux
                    width_overlap = wred - wbluenext

                    flux_int += width_nonoverlap * F[i, :]
                    flux_int += width_overlap * 0.5 * (F[i, :] + F[i + 1, :])

                    flux_int_err += width_nonoverlap * Ferr[i, :]
                    flux_int_err += width_overlap * 0.5 * np.sqrt(Ferr[i, :]**2 + Ferr[i + 1, :]**2)

                    idxlap = 1
            else:
                # --- last filter: no overlap with next possible ---
                width_last = wred - wblue
                flux_int += width_last * F[i, :]
                flux_int_err += width_last * Ferr[i, :]

            wred_prev = wred

        # Store results
        self.selected_filters_sorted = filters_sorted
        self.flux_int = flux_int
        self.flux_int_err = flux_int_err
        self._compute_luminosity_and_ni()
        
    def _compute_luminosity_and_ni(self):
        """
        Convert integrated flux to bolometric luminosity and estimate Ni mass.
        Stores results in self.lum_int, self.lum_int_err, self.L_peak, etc.
        """
        hdr = self.hdr
        flux_int = np.asarray(self.flux_int, dtype=float)
        flux_int_err = np.asarray(self.flux_int_err, dtype=float)
        t = np.asarray(self.time_grid, dtype=float)

        # ---------------------------------------------------------------
        # Convert integrated flux to luminosity: L = 4π d^2 F
        # distance in cm: 1 pc = 3.085677e18 cm
        # ---------------------------------------------------------------
        dist_cm = 3.085677e18 * 10.0 ** ((hdr.dmod + 5.0) / 5.0)
        dist_cm_err = dist_cm * np.log(10.0) * hdr.dmoderr / 5.0

        lum_int = 4.0 * np.pi * dist_cm**2 * flux_int

        # avoid division by zero in error term
        flux_ratio = np.where(flux_int > 0, flux_int_err / flux_int, 0.0)
        lum_int_err = lum_int * np.sqrt(
            2.0 * (dist_cm_err / dist_cm) ** 2 + flux_ratio**2
        )

        self.lum_int = lum_int
        self.lum_int_err = lum_int_err

        # ---------------------------------------------------------------
        # Estimate peak luminosity and nickel mass
        # ---------------------------------------------------------------
        i_peak = np.argmax(lum_int)
        L_peak = lum_int[i_peak]
        L_peak_err = lum_int_err[i_peak]
        t_peak = t[i_peak]

        M_ni, M_ni_err = estimate_56ni(
            L_peak=L_peak,
            L_peak_err=L_peak_err,

        )

        # store for later use in the GUI / file writer
        self.L_peak = L_peak
        self.L_peak_err = L_peak_err
        self.t_peak = t_peak
        self.M_ni = M_ni
        self.M_ni_err = M_ni_err
        
    def _build_step3(self):
        """Build the animation view (step 3)."""
        # Make sure fluxes & time_grid exist
        
        self._integrate_fluxes()
        if not hasattr(self, "interpflux") or not hasattr(self, "flux_int"):
            raise RuntimeError("Need interpflux and flux_int computed before step 3.")

        filters = self.selected_filters_sorted
        if not filters:
            raise RuntimeError("No filters selected for animation.")

        # Clear figure and reserve space for bottom buttons
        self._clear_step_widgets()
        self._clear_axes()
        self.fig.suptitle(
            "Step 3: Integrated flux animation\n"
            "Press 'Play'/'Pause' to control the animation.",
            y=0.99,
        )
        
        self.fig.subplots_adjust(bottom=0.18, hspace=0.35)

        # --- Axes layout: top = wavelength vs flux, bottom = integrated LC ---
        gs = self.fig.add_gridspec(
            nrows=2, ncols=1, height_ratios=[2.0, 1.0], hspace=0.25
        )
        ax_flux = self.fig.add_subplot(gs[0])
        ax_lc = self.fig.add_subplot(gs[1])

        self.ax_flux_step3 = ax_flux
        self.ax_lc_step3 = ax_lc

        # Convenience variables
        pb = self.pbinfo.loc[filters]
        lam_eff = pb["lambda_eff"].to_numpy()
        self.step3_lam_eff = lam_eff
        ew = pb["ew"].to_numpy()
        self.step3_ew = ew

        # Wavelength range (with small padding)
        w_min = np.min(lam_eff - ew / 2)
        w_max = np.max(lam_eff + ew / 2)
        dw = w_max - w_min
        ax_flux.set_xlim(w_min - 0.05 * dw, w_max + 0.05 * dw)

        # Y-range: max flux over all frames & filters
        F = self.interpflux  # shape (n_select, nt)
        self.step3_flux = F
        ymax = np.nanmax(F) * 1.1
        ax_flux.set_ylim(0.0, ymax)
        ax_flux.set_xlabel("Wavelength [\\AA]")
        ax_flux.set_ylabel("Flux")
        ax_flux.set_title("Flux in each filter \\& Integration Profile")

        # --- Create bar patches for each filter (one per filter) ---
        from matplotlib.patches import Rectangle

        self.step3_filters = filters  # store order used in animation
        self.step3_bar_patches = []

        for lam, width in zip(lam_eff, ew):
            x0 = lam - width / 2.0
            rect = Rectangle(
                (x0, 0.0),
                width,
                0.0,          # height set in frame update
                linewidth=0.0,
                edgecolor="none",
                facecolor="#083D77",
                alpha=0.7,
            )
            ax_flux.add_patch(rect)
            self.step3_bar_patches.append(rect)

        # --- Pink integration profile line ---
        (self.step3_profile_line,) = ax_flux.plot([], [], "-", lw=2.0, color="#DC0073", alpha = 0.7)

        # --- Integrated flux light curve panel ---
        t = self.time_grid
        fl_int = self.flux_int  # or whatever you call integrated L
        

        # Full LC, drawn once and never changed
        _ = ax_lc.scatter(t, fl_int, s = 15, color = "#083D77")
        ax_lc.set_xlabel("MJD")
        ax_lc.set_ylabel("Integrated Flux")  # or "Luminosity"

        # Moving point that will be updated per frame
        self.step3_lc_point = ax_lc.plot(
            [t[0]], [fl_int[0]], "o", markersize=10, color="#DC0073"
        )[0]

        # Optional: text label with current MJD / L
        self.step3_time_text = ax_lc.text(
            0.02, 0.90, f"MJD = {t[0]:.2f}", transform=ax_lc.transAxes
        )

        # --- Play / Pause buttons at bottom center ---
        # You likely already have Next / Back axes; reuse style/positions if you want.
      
        ax_play = self.fig.add_axes([self.x_back, 4*self.y_nav, self.nav_w, self.nav_h])
        ax_pause = self.fig.add_axes([self.x_next, 4*self.y_nav, self.nav_w, self.nav_h])

        from matplotlib.widgets import Button
        self.btn_play = Button(ax_play, "Play")
        self.btn_pause = Button(ax_pause, "Pause")

        self.btn_play.on_clicked(self._on_play_step3)
        self.btn_pause.on_clicked(self._on_pause_step3)

        # --- Animation state and timer ---
        self.step3_frame_index = 0
        self.step3_anim_running = False

        # timer in ms; ~10 fps -> 100 ms
        self.step3_timer = self.fig.canvas.new_timer(interval=100)
        self.step3_timer.add_callback(self._advance_frame_step3)

        # Draw first frame
        self._update_step3_frame(0)
        self.fig.canvas.draw_idle()
        

    def _update_step3_frame(self, i_frame: int):
        """Update all artists for given frame index."""
        filters = self.selected_filters_sorted
        if not filters:
            raise RuntimeError("No filters selected for animation frame update.")  
        pb = self.pbinfo.loc[filters]
        lam_eff = self.step3_lam_eff
        
        ew = self.step3_ew

        # Map filters (sorted by lambda) to rows in interpflux
        # interpflux shape = (n_select, nt), with row order = selected_filters
        selected = self.selected_filters
        idx_map = [selected.index(f) for f in filters]

        F = self.interpflux[idx_map, :]          # (n_filters, nt)
        flux_now = F[:, i_frame]                 # (n_filters,)

        # --- Update bar heights for each filter ---
        for rect, f_val in zip(self.step3_bar_patches, flux_now):
            rect.set_height(float(f_val))

        # --- Build pink integration profile as in IDL ---
        x_prof = []
        y_prof = []

        idxlap = 0  # 1 if current filter overlaps with previous
        y1prev = 0.0
        wred_prev = None

        nsel = len(filters)
        for ii in range(nsel):
            lam_i = lam_eff[ii]
            ew_i = ew[ii]
            wred = lam_i + ew_i / 2.0

            if ii > 0 and idxlap == 1 and wred_prev is not None:
                wblue = wred_prev
            else:
                wblue = lam_i - ew_i / 2.0

            if ii < nsel - 1:
                lam_next = lam_eff[ii + 1]
                ew_next = ew[ii + 1]
                wbluenext = lam_next - ew_next / 2.0

                if wred <= wbluenext:
                    # Isolated filter + gap
                    y0 = flux_now[ii]  # filter plateau
                    y1 = 0.5 * (flux_now[ii] + flux_now[ii + 1])  # gap midpoint

                    x_prof.extend([wblue, wblue, wred, wred, wbluenext])
                    y_prof.extend([y1prev, y0, y0, y1, y1])

                    idxlap = 0
                    y1prev = y1
                else:
                    # Overlap with next filter
                    y0 = flux_now[ii]
                    y1 = 0.5 * (flux_now[ii] + flux_now[ii + 1])

                    x_prof.extend([wblue, wblue, wbluenext, wbluenext, wred])
                    y_prof.extend([y1prev, y0, y0, y1, y1])

                    idxlap = 1
                    y1prev = y1
            else:
                # Last filter
                y0 = flux_now[ii]
                x_prof.extend([wblue, wblue, wred, wred])
                y_prof.extend([y1prev, y0, y0, 0.0])

            wred_prev = wred

        self.step3_profile_line.set_data(x_prof, y_prof)

        # --- Update LC point & label in the bottom panel ---
        t = self.time_grid
        L = self.flux_int

        t_now = t[i_frame]
        L_now = L[i_frame]

        # Move the single point
        self.step3_lc_point.set_data([t_now], [L_now])

        # Update label (optional)
        self.step3_time_text.set_text(f"MJD = {t_now:.2f}")
            
    def _advance_frame_step3(self):
        """Timer callback: advance one frame if running."""
        if not self.step3_anim_running:
            return

        nt = len(self.time_grid)
        # Wrap around at the end so Play restarts automatically
        self.step3_frame_index = (self.step3_frame_index + 1) % nt

        self._update_step3_frame(self.step3_frame_index)
        self.fig.canvas.draw_idle()


    def _on_play_step3(self, event):
        """Start or resume the animation."""
        if not hasattr(self, "step3_timer"):
            return
        self.step3_anim_running = True
        self.step3_timer.start()


    def _on_pause_step3(self, event):
        """Pause the animation."""
        if not hasattr(self, "step3_timer"):
            return
        self.step3_anim_running = False
        self.step3_timer.stop()
                
                
    def _build_step4(self):
        """
        Step 4: show the final bolometric light curve (luminosity vs time),
        print the peak luminosity and Ni mass, and offer to save the
        bolometric LC file and a PNG figure.
        """
        # Clear previous plots and step-specific widgets
        self._clear_step_widgets()
        self._clear_axes()

        # Make sure luminosity has been computed
        if not hasattr(self, "lum_int") or not hasattr(self, "lum_int_err"):
            # Fallback: recompute if needed
            self._integrate_fluxes()

        t = np.asarray(self.time_grid, dtype=float)
        L = np.asarray(self.lum_int, dtype=float)
        Lerr = np.asarray(self.lum_int_err, dtype=float)

        hdr = self.hdr

        # Figure title
        self.fig.suptitle(
            "Step 4: Bolometric Luminosity\n"
            "Final pseudo-bolometric light curve and Ni mass estimate",
            y=0.98,
        )

        # Single main axes
        ax = self.fig.add_axes([0.12, 0.25, 0.80, 0.65])
        self.axes_step4 = [ax]

        # Plot luminosity with shaded 1σ band
        ax.fill_between(
            t, L - Lerr, L + Lerr,
            alpha=0.2,
            color = "#BD2D87",
            linewidth=0,
        )
        ax.plot(t, L, "-", lw=1.5, color = "#BD2D87")

        ax.set_xlabel("MJD")
        ax.set_ylabel("Pseudobolometric luminosity [erg s$^{-1}$]")
        ax.grid(alpha=0.2)

        # Mark the peak
        i_peak = np.argmax(L)
        t_peak = t[i_peak]
        L_peak = L[i_peak]
        ax.plot(t_peak, L_peak, "o", ms=7)
        

        # Summary text box in the plot
        summary_lines = [
            rf"$L_{{\rm peak}} = {self.L_peak:.3e} \pm {self.L_peak_err:.3e}\ \mathrm{{erg\,s^{{-1}}}}$",
            rf"$M(^{56}\mathrm{{Ni}}) = {self.M_ni:.3f} \pm {self.M_ni_err:.3f}\ M_\odot$",
        ]
        ax.text(
            0.02, 0.97,
            "\n".join(summary_lines),
            transform=ax.transAxes,
            va="top",
            fontsize="large",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="none"),
        )

        # ------------------------------------------------------------------
        # Save buttons
        # ------------------------------------------------------------------
        btn_w = 0.16
        btn_h = 0.06
        y_btn = 0.12
        x_center = 0.5

        x_save_lc = x_center - btn_w - 0.03
        x_save_fig = x_center + 0.03

        ax_save_lc = self.fig.add_axes([x_save_lc, y_btn, btn_w, btn_h])
        ax_save_fig = self.fig.add_axes([x_save_fig, y_btn, btn_w, btn_h])

        self.btn_save_lc = Button(ax_save_lc, "Save LC file")
        self.btn_save_fig = Button(ax_save_fig, "Save figure")

        self.btn_save_lc.on_clicked(self._on_save_lc_step4)
        self.btn_save_fig.on_clicked(self._on_save_fig_step4)

        # Register these so _clear_step_widgets will remove them later
        if not hasattr(self, "step_widgets"):
            self.step_widgets = []
        self.step_widgets.extend([self.btn_save_lc, self.btn_save_fig])

        self.fig.canvas.draw_idle()
        
    def _on_save_lc_step4(self, event):
        """
        Write the bolometric light curve (time, L, Lerr) to a .dat file
        next to the input file, unless already overridden by user.
        """
        hdr = self.hdr
        t = np.asarray(self.time_grid, dtype=float)
        L = np.asarray(self.lum_int, dtype=float)
        Lerr = np.asarray(self.lum_int_err, dtype=float)

        infile = pathlib.Path(self.infile)
        # Build filter list / shortnames
        filters = self.selected_filters_sorted or self.filters
        filt_list = ", ".join(filters)
        shortnames = "".join(filter_shortname(f) for f in filters)

        # Default output name if none chosen elsewhere
        outname = f"{hdr.name}_lcbol_{shortnames}_py.dat"
        fout = infile.parent.parent / "test_results" / outname
        fout = pathlib.Path(fout)

        from datetime import datetime, UTC

        with fout.open("w") as f:
            f.write(f"#generated on {datetime.now(UTC).isoformat()} using single_sne bolometric LC wizard\n")
            f.write(f"#L_PEAK:   {self.L_peak:.8e} +/- {self.L_peak_err:.8e}\n")
            f.write(f"#MJD_PEAK: {self.t_peak:.5f}\n")
            f.write(f"#M56_NI:   {self.M_ni:.5f} +/- {self.M_ni_err}\n")
            f.write(f"#INFILE    {infile.name}\n")
            f.write(f"#NAME      {hdr.name}\n")
            f.write(f"#AV_HOST   {hdr.avhost:6.3f} +/- {hdr.avhosterr:6.3f}\n")
            f.write(f"#RV_HOST   {hdr.rvhost:6.3f} +/- {hdr.rvhosterr:6.3f}\n")
            f.write(f"#AV_MW     {hdr.avmw:6.3f} +/- {hdr.avmwerr:6.3f}\n")
            f.write(f"#RV_MW     {hdr.rvmw:6.3f} +/- {hdr.rvmwerr:6.3f}\n")
            f.write(f"#DIST_MOD  {hdr.dmod:6.3f} +/- {hdr.dmoderr:6.3f}\n")
            f.write(f"#NFILT     {len(filters):2d} ({filt_list})\n")
            f.write("#\n")
            f.write("#time[d]    lbol[erg/s]      lbolerr[erg/s]\n")
            for tt, LL, LLerr in zip(t, L, Lerr):
                f.write(f"{tt:10.3f}  {LL:15.8E}  {LLerr:15.8E}\n")

        print(f"Created file {fout}")
        
    def _on_save_fig_step4(self, event):
        """
        Save the current bolometric luminosity figure as a PNG next to infile.
        """
        hdr = self.hdr
        infile = pathlib.Path(self.infile)
        filters = self.selected_filters_sorted or self.filters
        shortnames = "".join(filter_shortname(f) for f in filters)

        outname = f"{hdr.name}_lcbol_{shortnames}_Lbol_py.png"
        ffig = infile.parent.parent / "test_results" / outname

        self.fig.savefig(ffig, dpi=150, bbox_inches="tight")
        print(f"Saved figure {ffig}")
            
    def run(self):
        """
        Start the GUI, block until the window is closed,
        and then return the final selections.
        """
        plt.show()  # blocks until plt.close(self.fig) is called somewhere

        # Clean up event connection(s) if they exist
        if hasattr(self, "cid_click_step1"):
            self.fig.canvas.mpl_disconnect(self.cid_click_step1)

        # If you store the final selection in self.selected_filters:
        if hasattr(self, "selected_filters"):
            return self.selected_filters

        # Fallback: derive directly from self.selected
        return [f for f, ok in self.selected.items() if ok]