from __future__ import annotations

import numpy as np
from dataclasses import dataclass
import pathlib

from typing import List, Optional, Sequence, Tuple
from scipy.interpolate import interp1d, UnivariateSpline
from pseudobolo.aux import al_av, estimate_56ni, filter_shortname, create_lc_df
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib
matplotlib.use('Qt5Agg') 

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
        integration_method: str = "histogram",
        debug: bool = False,
        save_steps: bool = False,
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
        integration_method: "histogram" or "trapezoidal"
            Determines how to integrate the fluxes
        debug : bool
            Extra prints, etc.
        save_steps: bool
            If turned on, it will also create .dat files containing the separate interpolations 
            for each filter, as well as a file with the integrated flux
        """
        self.debug = debug
        self.save_steps = save_steps
        self.integration_method = integration_method
        if (self.integration_method != "trapezoidal") and (self.integration_method != "histogram"):
            raise ValueError("Invalid integration method: Must be either 'histogram' or 'trapezoidal'") 
        
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
        self.RESULTS_DIR = pathlib.Path(__file__).resolve().parents[2] / "tests" / "tests_results"

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
    
    def _save_filter_interp(self, filter_name):
        """Store the GP interpolation results for a given filter."""
        from datetime import datetime
        from pytz import UTC
        hdr = self.hdr
        infile = self.infile
        time_grid = self.time_grid
        mag_interp = self.gp_interp[filter_name]["mag"]
        magerr_interp = self.gp_interp[filter_name]["err"]
        filters = self.selected_filters
        shortname = filter_shortname(filter_name)
        filt_list = ", ".join(filters)
        nmeas = len(mag_interp)
        interp_file = self.RESULTS_DIR / "filter_interp"/ f"{hdr.name}_{shortname}_interp_gp_py.dat"
        with interp_file.open("w") as f:
            f.write(f"#generated on {datetime.now(UTC).isoformat()} using single_sne bolometric LC wizard\n")
            f.write(f"#INFILE    {infile.name}\n")
            f.write(f"#NAME      {hdr.name}\n")
            f.write(f"#AV_HOST   {hdr.avhost:6.3f} +/- {hdr.avhosterr:6.3f}\n")
            f.write(f"#RV_HOST   {hdr.rvhost:6.3f} +/- {hdr.rvhosterr:6.3f}\n")
            f.write(f"#AV_MW     {hdr.avmw:6.3f} +/- {hdr.avmwerr:6.3f}\n")
            f.write(f"#RV_MW     {hdr.rvmw:6.3f} +/- {hdr.rvmwerr:6.3f}\n")
            f.write(f"#DIST_MOD  {hdr.dmod:6.3f} +/- {hdr.dmoderr:6.3f}\n")
            f.write(f"#NFILT     1 ({filter_name})\n")
            f.write("#\n")
            f.write("#time[mjd]    mag     mag_err\n")
            f.write(f"#FILTER {filter_name} - {nmeas} interpolated measurements\n")
            for tt, LL, LLerr in zip(time_grid, mag_interp, magerr_interp):
                f.write(f"{tt:10.3f}  {LL:15.8E}  {LLerr:15.8E}\n")

        print(f"Created file {interp_file}")
    
    
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
            self._clear_step_widgets()
            self._clear_axes()
            print("Returning to step 1.")
            self.btn_next.label.set_text("Next")
            self._build_step1()

        elif self.step == 3:
            self.step = 2
            print("Returning to step 2.")
            self._clear_step_widgets()
            self._clear_axes()
            self.btn_next.label.set_text("Next")  # or "Next" vs "Finish" logic later
            self._build_step2()
            
        elif self.step == 4:
            self.step = 3
            self._clear_step_widgets()
            self._clear_axes()
            print("Returning to step 3.")
            self.btn_next.label.set_text("Next")
            self._build_step3()


    def _build_step1(self):
        # make sure any old step widgets are gone
        self._clear_step_widgets()
        self._clear_axes()
        
        self.fig.suptitle("Step 1: Select filters\n"
                          "Click a panel to toggle selection (grey = OFF).\n"
                          "Press 'Next' when done.",
                          y=0.99,)

    
        # --- Determine filters and layout ---
        filters_unique = sorted(self.lightcurve_df["filter"].unique())
        
        # Build (filter, lambda_eff) list
        f_lam = []
        for f in filters_unique:
            lam_eff, ew = self._get_pbinfo(f) 
            f_lam.append((f, lam_eff))

        # Sort by lambda_eff
        f_lam_sorted = sorted(f_lam, key=lambda t: t[1])
        filters = [f for (f, _) in f_lam_sorted]
        self.filters = filters
        n_filt = len(filters)
        
        if n_filt == 0:
            raise ValueError("No filters found in lightcurve_df['filter'].")

        # --- 1. SQUARE GRID LOGIC ---
        # Calculate roughly square grid
        self.ncols = int(np.ceil(np.sqrt(n_filt)))
        self.nrows_filters = int(np.ceil(n_filt / self.ncols))
        
        self.preset_groups = {
                "UBVRI": ["U", "B", "V", "R", "I"],
                "BVRI":  ["B", "V", "R", "I"],
                "JHK":   ["J", "H", "K"],   
                "All":   ["*"],   
                "None": ["ZWXJ"],     
            }

        # --- UPDATED SPACING HERE ---
        self.gs = self.fig.add_gridspec(
            self.nrows_filters + 1,
            self.ncols,
            top=0.90,
            bottom=0.21,          
            left=0.08,
            right=0.98,
            height_ratios=[1.5] + [1.0] * self.nrows_filters,
            hspace=0.7,          # INCREASED: Vertical spacing between rows
            wspace=0.25,          # INCREASED: Horizontal spacing between cols
        )
        
        # --- Top coverage axis spanning all columns ---
        self.coverage_ax = self.fig.add_subplot(self.gs[0, :])
        self.coverage_ax.set_ylabel("Transmission")
        self.coverage_ax.grid(True, alpha = 0.1)
        self.coverage_ax.set_title("Filter Transmission Blocks", fontsize=10)
        self.coverage_ax.set_ylim(0, 1.2)
        # Only show wavelength label on top plot, push title up slightly
        self.coverage_ax.set_xlabel("Wavelength [$\\AA$]", labelpad=5) 

        # Track selection state and artists
        self.selected = {f: True for f in filters}
        self.coverage_patches = {}
        self.axes_step1 = []
        self.ax_to_filter = {}        

        # --- 2. COLOR SETUP ---
        import matplotlib.cm as cm
        # 'turbo' or 'nipy_spectral' are good for spectral progression
        cmap = cm.get_cmap('turbo') 
        
        # Store colors per filter
        filter_colors = {}

        # --- Plot transmission curves ---
        for i, f in enumerate(filters):
            lam_eff, ew = self._get_pbinfo(f)
            short = filter_shortname(f)
            lam_min = lam_eff - (ew/2.0)
            lam_max = lam_eff + (ew/2.0)
            wave = np.linspace(lam_min, lam_max, 50)
            
            # Pick color from map
            color = cmap(i / max(n_filt - 1, 1))
            filter_colors[f] = color
            
            # Plot with this color
            poly = self.coverage_ax.fill_between(wave, 0, 1, color=color, alpha=0.6, edgecolor='none')
            
            # Label with UPDATED ZORDER
            self.coverage_ax.annotate(f"{short}", xy=(lam_eff, 0.12), xycoords="data", 
                                      ha="center", fontsize="medium", rotation=90,
                                      zorder=200, fontweight = "bold", clip_on=False) # zorder=20 puts text on top
            self.coverage_patches[f] = poly

        # --- Per-filter subplots below ---
        idx = 0
        
        # Keep track of x-axis limits to sync them
        share_x_ax = None 

        for r in range(self.nrows_filters):
            for c in range(self.ncols):
                if idx >= n_filt:
                    break

                f = filters[idx]
                
                # Share X axis with the first plot to keep alignment perfect
                if share_x_ax is None:
                    ax = self.fig.add_subplot(self.gs[r + 1, c])
                    share_x_ax = ax
                else:
                    ax = self.fig.add_subplot(self.gs[r + 1, c], sharex=share_x_ax)

                self.axes_step1.append(ax)
                self.ax_to_filter[ax] = f

                sub = self.lightcurve_df[self.lightcurve_df["filter"] == f]

                # Use the SAME color as the top plot
                col = filter_colors[f]

                if "err" in sub.columns:
                    ax.errorbar(sub["mjd"], sub["mag"], yerr=sub["err"], 
                                fmt="o", ms=3, color=col, ecolor=col, alpha=0.9)
                else:
                    ax.plot(sub["mjd"], sub["mag"], "o", ms=3, color=col, alpha=0.9)

                # --- 3. OVERLAP PREVENTION ---
                # A. Put Title INSIDE the plot (using zorder ensures it sits on top of data)
                ax.text(0.5, 0.9, f, transform=ax.transAxes, 
                        ha='center', va='top', fontsize=9, fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1),
                        zorder=30)
                
                # B. Handle Axis Labels (Only on edges)
                is_bottom_row = (r == self.nrows_filters - 1)
                is_left_col = (c == 0)

                if is_bottom_row:
                    ax.set_xlabel("MJD", fontsize=9)
                else:
                    ax.tick_params(labelbottom=False) # Hide x ticks
                    
                if is_left_col:
                    ax.set_ylabel("Mag", fontsize=9)
                else:
                    ax.tick_params(labelleft=False) # Hide y ticks

                ax.invert_yaxis()
                ax.grid(True, alpha=0.1)

                idx += 1
                
        # Set limits for all (thanks to sharex, this applies to all)
        if self.axes_step1:
            self.axes_step1[0].set_xlim(self.mjd_min - 2, self.mjd_max + 2)

        # Hide any unused grid cells in the rectangle
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
        btn_w  = 0.13
        btn_h  = 0.05
        pad_x  = 0.02          
        y_presets = 0.1       

        total_w = 5 * btn_w + 3 * pad_x
        x0 = 0.5 - total_w / 2  

        ax_ubvri = self.fig.add_axes([x0 + 0*(btn_w+pad_x), y_presets, btn_w, btn_h])
        ax_bvri  = self.fig.add_axes([x0 + 1*(btn_w+pad_x), y_presets, btn_w, btn_h])
        ax_jhk   = self.fig.add_axes([x0 + 2*(btn_w+pad_x), y_presets, btn_w, btn_h])
        ax_all   = self.fig.add_axes([x0 + 3*(btn_w+pad_x), y_presets, btn_w, btn_h])
        ax_none  = self.fig.add_axes([x0 + 4*(btn_w+pad_x), y_presets, btn_w, btn_h])

        self.btn_ubvri = Button(ax_ubvri, "UBVRI")
        self.btn_bvri  = Button(ax_bvri,  "BVRI")
        self.btn_jhk   = Button(ax_jhk,   "JHK")
        self.btn_all   = Button(ax_all,   "All")
        self.btn_none = Button(ax_none, "None")

        self.btn_ubvri.on_clicked(self._make_preset_callback("UBVRI"))
        self.btn_bvri.on_clicked(self._make_preset_callback("BVRI"))
        self.btn_jhk.on_clicked(self._make_preset_callback("JHK"))
        self.btn_all.on_clicked(self._make_preset_callback("All"))
        self.btn_none.on_clicked(self._make_preset_callback("None"))
        
        self._step_widgets = [
            self.btn_ubvri,
            self.btn_bvri,
            self.btn_jhk,
            self.btn_all,
            self.btn_none
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
        filters = self.selected_filters
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
                    
                    if self.save_steps:
                        self._save_filter_interp(f)

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
            ax.set_xlim(min(self.time_grid) - 2, max(self.time_grid) + 2)

        # Hide any unused cells
        total_cells = nrows * self.ncols
        for extra in range(idx, total_cells):
            r = extra // self.ncols
            c = extra % self.ncols
            ax = self.fig.add_subplot(self.gs2[r, c])
            ax.set_visible(False)

        self.fig.subplots_adjust(bottom=0.18, top=0.88, hspace=0.4)
        self.fig.canvas.draw_idle()
        # 1. Define a filename (maybe include the filter name or step number)
        filt_list = ", ".join(filters)
        filename = self.RESULTS_DIR /f"{self.hdr.name}_filter_interpolations_{filt_list}.png"
        # 2. Save the figure
        self.fig.savefig(filename, dpi=300, bbox_inches='tight')

        print(f"Saved figure to {filename}")
        
        
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
        
    def _integrate_fluxes_trapezoidal(self):
        self._compute_fluxes_from_gp()
        
        filters = self.selected_filters
        if not filters:
            raise RuntimeError("No filters selected.")

        # 1. Sort filters by wavelength
        filters_sorted = sorted(filters, key=lambda f: self.pbinfo.loc[f, "lambda_eff"])
        self.selected_filters_sorted = filters_sorted
        
        # 2. Get Effective Wavelengths and Widths
        lam_effs = np.array([float(self.pbinfo.loc[f, "lambda_eff"]) for f in filters_sorted])
        ews      = np.array([float(self.pbinfo.loc[f, "ew"]) for f in filters_sorted])
        
        # 3. Get Fluxes (n_filters, n_time_steps)
        idx_map = [filters.index(f) for f in filters_sorted]
        F = self.interpflux[idx_map, :]    # Flux
        Ferr = self.interpfluxerr[idx_map, :] # Error
        
        # 4. Define the Integration Grid
        # We start at the Blue Edge of the first filter
        # We go through the Centers of all filters
        # We end at the Red Edge of the last filter
        
        start_lam = lam_effs[0] - ews[0]/2.0
        end_lam   = lam_effs[-1] + ews[-1]/2.0
        
        # 5. Perform Integration per timestep
        nt = F.shape[1]
        flux_int = np.zeros(nt)
        flux_int_err = np.zeros(nt)

        for t in range(nt):
            # Construct the SED curve for this specific timestep
            fluxes_at_t = F[:, t]
            errs_at_t   = Ferr[:, t]
            
            # --- The Integration Arrays ---
            # X: [BlueEdge_1,  Center_1,    Center_2,   ..., Center_N,    RedEdge_N]
            # Y: [Flux_1,      Flux_1,      Flux_2,     ..., Flux_N,      Flux_N   ]
            
            # We assume flux is constant from the Blue Edge to the first Center
            # We linearly interpolate between Centers
            # We assume flux is constant from the last Center to the Red Edge
            
            x_grid = np.concatenate(([start_lam], lam_effs, [end_lam]))
            y_grid = np.concatenate(([fluxes_at_t[0]], fluxes_at_t, [fluxes_at_t[-1]]))
            
            # Trapezoidal Integration (Area under the curve)
            flux_int[t] = np.trapz(y_grid, x_grid)
            
            # Error Propagation (Quadrature for Trapezoidal Rule is complex, 
            # approximate by treating segments as independent blocks or linear combos)
            # Simple approx: Integrate variance similarly
            # (Strictly speaking, you should sum (0.5 * dx * err)^2, but this is a close proxy)
            y_err_grid = np.concatenate(([errs_at_t[0]], errs_at_t, [errs_at_t[-1]]))
            # Square the errors, integrate squared errors, take sqrt? 
            # A simplified approach usually used in mklcbol versions:
            # Sum of (width * error) in quadrature.
            
            # Let's stick to a robust approximation for errors:
            # We calculate the widths associated with each filter
            # For filter i: width = (lam[i+1] - lam[i-1]) / 2
            
            # Define "effective integration width" for each filter in the chain
            dlam = np.zeros_like(lam_effs)
            dlam[0] = (lam_effs[1] - lam_effs[0])/2.0 + ews[0]/2.0 # First filter
            dlam[-1] = (lam_effs[-1] - lam_effs[-2])/2.0 + ews[-1]/2.0 # Last filter
            dlam[1:-1] = (lam_effs[2:] - lam_effs[:-2]) / 2.0 # Middle filters
            
            flux_int_err[t] = np.sqrt(np.sum((dlam * errs_at_t)**2))

        self.flux_int = flux_int
        self.flux_int_err = flux_int_err

    def _integrate_fluxes_histogram(self):
        self._compute_fluxes_from_gp()
        filters = self.selected_filters
        
        # 1. Sort filters by wavelength
        filters_sorted = sorted(filters, key=lambda f: self.pbinfo.loc[f, "lambda_eff"])
        
        # 2. Get Fluxes: Shape should be (n_filters, n_timesteps)
        idx_map = [filters.index(f) for f in filters_sorted]
        F = self.interpflux[idx_map, :]    
        Ferr = self.interpfluxerr[idx_map, :] 
        
        n_filters, nt = F.shape # e.g. (5 filters, 100 timesteps)

        # 3. Define all Edges
        lam_effs = np.array([float(self.pbinfo.loc[f, "lambda_eff"]) for f in filters_sorted])
        ews      = np.array([float(self.pbinfo.loc[f, "ew"]) for f in filters_sorted])
        blue_edges = lam_effs - ews/2.0
        red_edges  = lam_effs + ews/2.0
        
        # 4. Create Unique Boundaries Grid
        boundaries = np.unique(np.concatenate([blue_edges, red_edges]))
        boundaries.sort()
        
        # 5. Initialize Output Array (Vector of size nt)
        # CRITICAL: Do not initialize as scalar 0.0
        flux_int = np.zeros(nt, dtype=float)
        flux_int_err_sq = np.zeros(nt, dtype=float) # Sum of variances

        # 6. Loop over WAVELENGTH bins
        for i in range(len(boundaries) - 1):
            w_start = boundaries[i]
            w_end   = boundaries[i+1]
            w_center = (w_start + w_end) / 2.0
            width = w_end - w_start
            
            active_mask = (blue_edges <= w_center) & (red_edges >= w_center)
            
            if np.any(active_mask):
                # --- Normal Overlap Logic (Same as before) ---
                avg_flux_in_bin = np.mean(F[active_mask, :], axis=0)
                n_active = np.sum(active_mask)
                sum_sq_errs = np.sum(Ferr[active_mask, :]**2, axis=0)
                var_mean_in_bin = sum_sq_errs / (n_active**2)
                
                flux_int += width * avg_flux_in_bin
                flux_int_err_sq += (width**2) * var_mean_in_bin

            else:
                # --- GAP LOGIC (New) ---
                # Find neighbors
                idx_left = np.where(np.isclose(red_edges, w_start))[0]
                idx_right = np.where(np.isclose(blue_edges, w_end))[0]
                
                flux_neighbors = []
                var_neighbors = []
                
                if idx_left.size > 0:
                    # Mean of filters ending on the left
                    flux_neighbors.append(np.mean(F[idx_left, :], axis=0))
                    # Var of mean
                    v_left = np.sum(Ferr[idx_left, :]**2, axis=0) / (idx_left.size**2)
                    var_neighbors.append(v_left)

                if idx_right.size > 0:
                    # Mean of filters starting on the right
                    flux_neighbors.append(np.mean(F[idx_right, :], axis=0))
                    # Var of mean
                    v_right = np.sum(Ferr[idx_right, :]**2, axis=0) / (idx_right.size**2)
                    var_neighbors.append(v_right)
                
                if flux_neighbors:
                    # Average the Left Group and Right Group
                    # (This mimics the IDL logic: 0.5 * (F_left + F_right))
                    avg_gap_flux = np.mean(flux_neighbors, axis=0)
                    
                    # Error prop: Sqrt(Sum(Vars)) / N
                    # We have N groups (usually 2: left and right)
                    n_groups = len(flux_neighbors)
                    sum_vars = np.sum(var_neighbors, axis=0)
                    var_gap = sum_vars / (n_groups**2)
                    
                    flux_int += width * avg_gap_flux
                    flux_int_err_sq += (width**2) * var_gap

        # Finalize
        self.flux_int = flux_int
        self.flux_int_err = np.sqrt(flux_int_err_sq)
        self.selected_filters_sorted = filters_sorted


                
    def _integrate_fluxes(self):
        """
        Integrate the fluxes over wavelength, handling gaps and overlaps,
        following the original IDL mklcbol logic.
        Populates:
          - self.flux      : (nt,) bolometric luminosity
          - self.flux_int_err  : (nt,) bolometric luminosity error
        """
        if self.integration_method == "trapezoidal":
            self._integrate_fluxes_trapezoidal()
        else:
            self._integrate_fluxes_histogram()
        
        if self.save_steps:
            infile = self.infile
            hdr = self.hdr
            flux_int = self.flux_int
            flux_int_err = self.flux_int_err
            filters = self.selected_filters_sorted or self.filters
            filt_list = ", ".join(filters)
            shortnames = "".join(filter_shortname(f) for f in filters)
            flux_peak = np.max(flux_int)
            flux_peak_err = flux_int_err[np.argmax(flux_int)]
            t_peak = self.time_grid[np.argmax(flux_int)]
            t = self.time_grid
            #Save integrated flux file for comparison:
            # Default output name if none chosen elsewhere
            outname = f"{hdr.name}_lcbol_{shortnames}_flux_int_py.dat"
            output_dir = infile.parent.parent / "tests_results"
            output_dir.mkdir(parents=True, exist_ok=True)
            fout = output_dir / outname
            fout = pathlib.Path(fout)

            from datetime import datetime, UTC

            with fout.open("w") as f:
                f.write(f"#generated on {datetime.now(UTC).isoformat()} using single_sne bolometric LC wizard\n")
                f.write(f"#FLUX_INT_PEAK:   {flux_peak:.8e} +/- {flux_peak_err:.8e}\n")
                f.write(f"#MJD_PEAK: {t_peak:.5f}\n")
                f.write(f"#INFILE    {infile.name}\n")
                f.write(f"#NAME      {hdr.name}\n")
                f.write(f"#AV_HOST   {hdr.avhost:6.3f} +/- {hdr.avhosterr:6.3f}\n")
                f.write(f"#RV_HOST   {hdr.rvhost:6.3f} +/- {hdr.rvhosterr:6.3f}\n")
                f.write(f"#AV_MW     {hdr.avmw:6.3f} +/- {hdr.avmwerr:6.3f}\n")
                f.write(f"#RV_MW     {hdr.rvmw:6.3f} +/- {hdr.rvmwerr:6.3f}\n")
                f.write(f"#DIST_MOD  {hdr.dmod:6.3f} +/- {hdr.dmoderr:6.3f}\n")
                f.write(f"#NFILT     {len(filters):2d} ({filt_list})\n")
                f.write("#\n")
                f.write("#time[d]    flux_int     flux_int_err\n")
                for tt, LL, LLerr in zip(t, flux_int, flux_int_err):
                    f.write(f"{tt:10.3f}  {LL:15.8E}  {LLerr:15.8E}\n")

            print(f"Created file {fout}")
            
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
        
        # Ensure integration method is set (Default to trapezoidal if missing)
        if not hasattr(self, "integration_method"):
            self.integration_method = "trapezoidal"

        self._integrate_fluxes()
        
        if not hasattr(self, "interpflux") or not hasattr(self, "flux_int"):
            raise RuntimeError("Need interpflux and flux_int computed before step 3.")

        filters = self.selected_filters_sorted
        if not filters:
            raise RuntimeError("No filters selected for animation.")

        # Clear figure and reserve space for bottom buttons
        self._clear_step_widgets()
        self._clear_axes()
        
        method_title = "Trapezoidal" if self.integration_method == "trapezoidal" else "Merged Histogram"
        self.fig.suptitle(
            f"Step 3: Integrated flux animation ({method_title})\n"
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
        ax_flux.set_title("Flux in each filter \& Integration Profile")

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
                alpha=0.4, # Made slightly more transparent to see the pink line better
            )
            ax_flux.add_patch(rect)
            self.step3_bar_patches.append(rect)

        # --- Pink integration profile line ---
        (self.step3_profile_line,) = ax_flux.plot([], [], "-", lw=2.5, color="#DC0073", alpha = 0.9)

        # --- Integrated flux light curve panel ---
        t = self.time_grid
        fl_int = self.flux_int 

        # Full LC, drawn once and never changed
        _ = ax_lc.scatter(t, fl_int, s = 15, color = "#083D77")
        ax_lc.set_xlabel("MJD")
        ax_lc.set_ylabel("Integrated Flux")

        # Moving point that will be updated per frame
        self.step3_lc_point = ax_lc.plot(
            [t[0]], [fl_int[0]], "o", markersize=10, color="#DC0073"
        )[0]

        # Optional: text label with current MJD / L
        self.step3_time_text = ax_lc.text(
            0.02, 0.90, f"MJD = {t[0]:.2f}", transform=ax_lc.transAxes
        )

        # --- Play / Pause buttons at bottom center ---
        ax_play = self.fig.add_axes([self.x_back, 4*self.y_nav, self.nav_w, self.nav_h])
        ax_pause = self.fig.add_axes([self.x_next, 4*self.y_nav, self.nav_w, self.nav_h])

        from matplotlib.widgets import Button
        self.btn_play = Button(ax_play, "Play")
        self.btn_pause = Button(ax_pause, "Pause")

        self.btn_play.on_clicked(self._on_play_step3)
        self.btn_pause.on_clicked(self._on_pause_step3)
        self._step_widgets = [
            self.btn_play, self.btn_pause
        ]

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
        """Update all artists for given frame index using selected Integration Method."""
        filters = self.selected_filters_sorted
        if not filters:
            return 
            
        lam_eff = self.step3_lam_eff
        ew = self.step3_ew

        # Map filters (sorted by lambda) to rows in interpflux
        selected = self.selected_filters
        idx_map = [selected.index(f) for f in filters]

        F = self.interpflux[idx_map, :]          # (n_filters, nt)
        flux_now = F[:, i_frame]                 # (n_filters,)

        # --- Update bar heights for each filter ---
        for rect, f_val in zip(self.step3_bar_patches, flux_now):
            rect.set_height(float(f_val))

        # --- Build pink integration profile based on Method ---
        x_prof = []
        y_prof = []
        
        # Determine edges for current frame
        blue_edges = lam_eff - ew / 2.0
        red_edges = lam_eff + ew / 2.0

        if self.integration_method == "trapezoidal":
            # --- METHOD A: TRAPEZOIDAL ---
            # Connect centers, extend flat to outer edges
            
            # 1. Start at Blue Edge of Filter 0
            # 2. Go through all centers
            # 3. End at Red Edge of Filter -1
            
            x_prof = np.concatenate(([blue_edges[0]], lam_eff, [red_edges[-1]]))
            y_prof = np.concatenate(([flux_now[0]], flux_now, [flux_now[-1]]))

        else:
            # --- METHOD B: HISTOGRAM (Merged Bins) ---
            
            # 1. Create unique sorted boundaries from all edges
            boundaries = np.unique(np.concatenate([blue_edges, red_edges]))
            boundaries.sort()
            
            # 2. Iterate through bins
            for i in range(len(boundaries) - 1):
                w_start = boundaries[i]
                w_end = boundaries[i+1]
                w_center = (w_start + w_end) / 2.0
                
                # Check which filters are active in this bin
                active_mask = (blue_edges <= w_center) & (red_edges >= w_center)
                
                if np.any(active_mask):
                    # Normal Overlap: Mean of active filters
                    val = np.mean(flux_now[active_mask])
                else:
                    # --- GAP LOGIC: Bridge Neighbors ---
                    # Find filters that END exactly at w_start
                    left_neighbors = np.where(np.isclose(red_edges, w_start))[0]
                    # Find filters that START exactly at w_end
                    right_neighbors = np.where(np.isclose(blue_edges, w_end))[0]
                    
                    neighbor_vals = []
                    
                    if left_neighbors.size > 0:
                        neighbor_vals.append(np.mean(flux_now[left_neighbors]))
                    if right_neighbors.size > 0:
                        neighbor_vals.append(np.mean(flux_now[right_neighbors]))
                        
                    if neighbor_vals:
                        # Mean of the Left and Right neighbors
                        val = np.mean(neighbor_vals)
                    else:
                        val = 0.0 
                
                x_prof.extend([w_start, w_end])
                y_prof.extend([val, val])

        self.step3_profile_line.set_data(x_prof, y_prof)

        # --- Update LC point & label in the bottom panel ---
        t = self.time_grid
        L = self.flux_int

        t_now = t[i_frame]
        L_now = L[i_frame]

        # Move the single point
        self.step3_lc_point.set_data([t_now], [L_now])

        # Update label
        self.step3_time_text.set_text(f"MJD = {t_now:.2f}")

    def _advance_frame_step3(self):
        """Timer callback: advance one frame if running."""
        if not self.step3_anim_running:
            return

        nt = len(self.time_grid)
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
        fout = infile.parent.parent / "tests_results" / outname
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
        ffig = infile.parent.parent / "tests_results" / outname

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