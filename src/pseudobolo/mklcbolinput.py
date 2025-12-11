#!/usr/bin/env python
"""
Build lcbol input files for mklcbol from BlackGEM + Swift UVOT photometry.

Output format matches the IDL / legacy mklcbolinput.pro style, e.g.:

#generated on ...
#
#NAME     sn2002bo
#AV_HOST  0.850 +/- 0.113
#RV_HOST  2.190 +/- 0.290
#AV_MW    0.077 +/- 0.000
#RV_MW    3.100 +/- 0.000
#DIST_MOD 31.900 +/- 0.200
#NFILT    8 (B_Bessell,H_2MASS,...)
#
#FILTER B_Bessell - 50 MEASUREMENTS (MJD MAG MAGERR)
...

For SN2025cy we’ll typically combine:
- BlackGEM: u_BG, q_BG, i_BG
- Swift UVOT AB: uu_Swift, bb_Swift, vv_Swift, uvw1_Swift, uvw2_Swift, uvm2_Swift
"""

from __future__ import annotations

from dataclasses import dataclass
import pathlib
from typing import Optional, Sequence

from datetime import datetime, UTC
import numpy as np
import pandas as pd


# -------------------------------
# Filter name mappings
# -------------------------------

# BlackGEM filter mapping: internal -> pbinfo name
BG_FILTER_MAP = {
    "u": "u_BG",
    "g": "g_BG",
    "q": "q_BG",
    "r": "r_BG",
    "i": "i_BG",
    "z": "z_BG",
}

@dataclass
class ExtinctionInfo:
    """Simple container for extinction / distance info used in the lcbol header."""
    av_host: float = 0.0
    av_host_err: float = 0.0
    rv_host: float = 3.1
    rv_host_err: float = 0.0

    av_mw: float = 0.0
    av_mw_err: float = 0.0
    rv_mw: float = 3.1
    rv_mw_err: float = 0.0

    dist_mod: float = 31.9
    dist_mod_err: float = 0.2


# -----------------------------------
# Utility: read BlackGEM photometry
# -----------------------------------

def read_blackgem_csv(path: str | pathlib.Path) -> pd.DataFrame:
    """
    Read BlackGEM photometry from a CSV file with columns:

        MJD, filter, AB_mag, AB_mag_err

    and return a standardized DataFrame with columns:
        ['mjd', 'mag', 'magerr', 'pb_name']

    Magnitudes are in AB.
    """
    path = pathlib.Path(path)
    df = pd.read_csv(path)

    # make sure names are exactly what we expect
    expected = {"MJD", "filter", "AB_mag", "AB_mag_err"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in BlackGEM file {path}: {missing}")

    # normalize + map to pbinfo names
    filt = df["filter"].astype(str).str.strip()
    pb_name = filt.map(BG_FILTER_MAP)

    # drop filters we don't know how to map
    mask = pb_name.notna()
    if not mask.any():
        raise ValueError(f"No BlackGEM filters could be mapped using BG_FILTER_MAP in {path}")

    out = pd.DataFrame(
        {
            "mjd": df.loc[mask, "MJD"].astype(float),
            "mag": df.loc[mask, "AB_mag"].astype(float),
            "magerr": df.loc[mask, "AB_mag_err"].astype(float),
            "pb_name": pb_name[mask],
        }
    )

    return out


#UVOT Filter mapping
UVOT_FILTER_MAP = {
    "u": "uu_Swift",
    "b": "bb_Swift",
    "v": "vv_Swift",
    "uvw1": "uvw1_Swift",
    "uvw2": "uvw2_Swift",
    "uvm2": "uvm2_Swift",
}

def read_uvot_ab_txt(path: str | pathlib.Path) -> pd.DataFrame:
    """
    Read Swift UVOT photometry (already in AB mags) from a text file with columns:

        filter  MJD  AB_MAG  AB_MAG_ERR

    Returns a DataFrame with columns:
        ['mjd', 'mag', 'magerr', 'pb_name']
    """
    path = pathlib.Path(path)

    df = pd.read_csv(
        path,
        sep = r"\s+",
        comment="#",
        header=0,
    )

    expected = {"filter", "MJD", "AB_MAG", "AB_MAG_ERR"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in UVOT file {path}: {missing}")

    # Normalize filter names and map to pbinfo names
    filt = df["filter"].astype(str).str.strip().str.lower()
    pb_name = filt.map(UVOT_FILTER_MAP)

    mask = pb_name.notna()
    if not mask.any():
        raise ValueError(f"No UVOT filters could be mapped using UVOT_FILTER_MAP in {path}")

    out = pd.DataFrame(
        {
            "mjd": df.loc[mask, "MJD"].astype(float),
            "mag": df.loc[mask, "AB_MAG"].astype(float),
            "magerr": df.loc[mask, "AB_MAG_ERR"].astype(float),
            "pb_name": pb_name[mask],
        }
    )

    return out


def get_ned_ebv(name: str) -> Optional[float]:
    """
    Placeholder for a NED E(B-V) query.

    In *your* local environment you can implement this using `requests`
    + `BeautifulSoup` to scrape the 'Galactic Extinction E(B-V)' value
    from the NED object page.

    For now this function just returns None.
    """
    # You can implement something like:
    #   import requests
    #   from bs4 import BeautifulSoup
    #
    #   url = f"https://ned.ipac.caltech.edu/cgi-bin/objsearch?objname={name}&extend=no&..."
    #   html = requests.get(url, timeout=10).text
    #   ... parse E(B-V) ...
    #
    # For now:
    return None


# -----------------------------------
# Main builder: combine data & write file
# -----------------------------------

def build_mklcbol_input(
    outfile: str | pathlib.Path,
    sn_name: str,
    bg_csv: str | pathlib.Path | None = None,
    uvot_txt: str | pathlib.Path | None = None,
    extinction: Optional[ExtinctionInfo] = None,
    ned_name: Optional[str] = None,
) -> pathlib.Path:
    """
    Build an mklcbol-compatible input file for `mklcbol` from
    BlackGEM and Swift UVOT AB photometry.

    Parameters
    ----------
    outfile : str or Path
        Output path for the lcbol input file.
    sn_name : str
        Supernova name, e.g. 'SN2025cy'.
    bg_csv : path-like, optional
        BlackGEM photometry CSV file (AB mags).
    uvot_txt : path-like, optional
        Swift UVOT AB photometry text file.
    extinction : ExtinctionInfo, optional
        Extinction & distance info for header. If None, a default
        ExtinctionInfo() is used (you should override in practice).
    ned_name : str, optional
        If given and extinction.av_mw == 0, you *could* use this
        together with get_ned_ebv() to fill AV_MW = 3.1 * E(B-V).

    Returns
    -------
    Path
        Path of the created file.
    """
    outfile = pathlib.Path(outfile)

    if extinction is None:
        extinction = ExtinctionInfo()

    # Optionally fetch MW extinction from NED
    if extinction.av_mw == 0.0 and ned_name is not None:
        ebv = get_ned_ebv(ned_name)
        if ebv is not None:
            extinction.av_mw = 3.1 * ebv

    # -------------------------
    # Read + standardize data
    # -------------------------
    frames: list[pd.DataFrame] = []

    if bg_csv is not None:
        bg = read_blackgem_csv(bg_csv)
        frames.append(bg[["mjd", "mag", "magerr", "pb_name"]])

    if uvot_txt is not None:
        uv = read_uvot_ab_txt(uvot_txt)
        frames.append(uv[["mjd", "mag", "magerr", "pb_name"]])

    if not frames:
        raise RuntimeError("No photometry provided (both bg_csv and uvot_txt are None or empty).")

    phot = pd.concat(frames, ignore_index=True)

    # Drop NaNs, sort by filter + mjd
    phot = phot.dropna(subset=["mjd", "mag", "magerr", "pb_name"])
    phot = phot.sort_values(by=["pb_name", "mjd"]).reset_index(drop=True)

    # -------------------------
    # Group by filter
    # -------------------------
    filters = phot["pb_name"].unique()
    nfilt = len(filters)

    # For NFILT header, we want the list in parentheses
    filters_sorted = sorted(filters)

    # -------------------------
    # Write output file
    # -------------------------
    now = datetime.now(UTC).strftime("%a %b %d %H:%M:%S %Y")


    with outfile.open("w") as f:
        f.write(f"#generated on {now} using mklcbolinput.py\n")
        f.write("#\n")
        f.write(f"#NAME     {sn_name}\n")
        f.write(
            f"#AV_HOST  {extinction.av_host:6.3f} +/- {extinction.av_host_err:6.3f}\n"
        )
        f.write(
            f"#RV_HOST  {extinction.rv_host:6.3f} +/- {extinction.rv_host_err:6.3f}\n"
        )
        f.write(
            f"#AV_MW    {extinction.av_mw:6.3f} +/- {extinction.av_mw_err:6.3f}\n"
        )
        f.write(
            f"#RV_MW    {extinction.rv_mw:6.3f} +/- {extinction.rv_mw_err:6.3f}\n"
        )
        f.write(
            f"#DIST_MOD {extinction.dist_mod:6.3f} +/- {extinction.dist_mod_err:6.3f}\n"
        )
        filt_list_str = ",".join(filters_sorted)
        f.write(f"#NFILT    {nfilt:d} ({filt_list_str})\n")
        f.write("#\n")

        # For each filter, in sorted order
        for pb_name in filters_sorted:
            sub = phot[phot["pb_name"] == pb_name].copy()
            nmeas = len(sub)
            f.write(f"#FILTER {pb_name} - {nmeas:d} MEASUREMENTS (MJD MAG MAGERR)\n")
            for _, row in sub.iterrows():
                f.write(
                    f"{row['mjd']:10.4f}  {row['mag']:9.3f}  {row['magerr']:8.3f}\n"
                )

    print(f"[ INFO  ] Created lcbol input file: {outfile}")
    print(f"[ INFO  ] Filters included: {', '.join(filters_sorted)}")

    return outfile


# -----------------------------------
# Optional CLI
# -----------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build lcbol input file for mklcbol from BlackGEM + Swift UVOT photometry."
    )
    parser.add_argument("outfile", help="Output lcbol input file (e.g. SN2025cy_lcbolinput.dat)")
    parser.add_argument("--name", required=True, help="SN name, e.g. SN2025cy")
    parser.add_argument("--bg-csv", help="BlackGEM photometry CSV (AB mags)")
    parser.add_argument("--uvot-txt", help="Swift UVOT AB photometry text file")
    parser.add_argument("--av-host", type=float, default=0.0)
    parser.add_argument("--av-host-err", type=float, default=0.0)
    parser.add_argument("--rv-host", type=float, default=3.1)
    parser.add_argument("--rv-host-err", type=float, default=0.0)
    parser.add_argument("--av-mw", type=float, default=0.0)
    parser.add_argument("--av-mw-err", type=float, default=0.0)
    parser.add_argument("--rv-mw", type=float, default=3.1)
    parser.add_argument("--rv-mw-err", type=float, default=0.0)
    parser.add_argument("--dist-mod", type=float, default=31.9)
    parser.add_argument("--dist-mod-err", type=float, default=0.2)
    parser.add_argument("--ned-name", help="NED object name to optionally fetch E(B-V) for MW extinction")

    args = parser.parse_args()

    ext = ExtinctionInfo(
        av_host=args.av_host,
        av_host_err=args.av_host_err,
        rv_host=args.rv_host,
        rv_host_err=args.rv_host_err,
        av_mw=args.av_mw,
        av_mw_err=args.av_mw_err,
        rv_mw=args.rv_mw,
        rv_mw_err=args.rv_mw_err,
        dist_mod=args.dist_mod,
        dist_mod_err=args.dist_mod_err,
    )

    build_mklcbol_input(
        outfile=args.outfile,
        sn_name=args.name,
        bg_csv=args.bg_csv,
        uvot_txt=args.uvot_txt,
        extinction=ext,
        ned_name=args.ned_name,
    )