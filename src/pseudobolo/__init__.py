"""
pseudobolo: interactive tool to build pseudo-bolometric light curves.

Main user-facing objects:
- create_lc_df: parse lcbolinput + pbinfo into DataFrames
- BolLCWizard: interactive Matplotlib GUI to build the bolometric LC
"""

from .aux import create_lc_df
from .make_bol_lc import BolLCWizard

__all__ = ["create_lc_df", "PseudoBoloWizard"]