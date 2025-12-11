from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

#--------------------------------------------------
# Attempt at translating Stéphane's IDL code into Python, started 19.11.25
#----------------------------------------------------------------


@dataclass
class LightCurveHeader:
    name: str
    nfilt: int
    # extinction / distance info
    avhost: float
    avhosterr: float
    rvhost: float
    rvhosterr: float
    avmw: float
    avmwerr: float
    rvmw: float
    rvmwerr: float
    dmod: float
    dmoderr: float


@dataclass
class FilterLightCurve:
    """One filter's light curve (IDL lcdata[i])."""
    filt: str
    time: np.ndarray  # days
    mag: np.ndarray
    magerr: np.ndarray


@dataclass
class PassbandInfo:
    """One row from pbinfo.dat."""
    name: str
    lambda_eff: float  # effective wavelength (Å)
    ew: float          # equivalent width (Å)
    zpt: float         # zeropoint (mag)

