import pathlib 
import sys
import numpy as np


# Add ../src to sys.path
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pseudobolo.make_bol_lc import PseudoBoloWizard

DATA_DIR = ROOT / "tests" / "tests_data"
RESULTS_DIR = ROOT / "tests" / "tests_results"

def main():
    base = pathlib.Path(__file__).resolve().parents[1]  # repo root (single_sne level)
    infile = base / "tests" / "tests_data" / "sn2002bo_lcbolinput.dat"
    pbinfo_file = base / "tests" / "tests_data" / "pbinfo.dat"

    wizard = PseudoBoloWizard(infile, pbinfo_file)
    selected_filters = wizard.run()
    
    
    #compared saved results to IDL outputs:
    data_idl = DATA_DIR / "sn2002bo_lcbol_UBVRI.dat"
    t_idl, Lbol_idl, Lbolerr_idl = np.loadtxt(
    data_idl,
    comments="#",
    unpack=True,      # <- THIS is the key
    usecols=(0, 1, 2) # optional, but makes it explicit
    )


if __name__ == "__main__":
    main()