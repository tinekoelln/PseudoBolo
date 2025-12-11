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
    infile_2025cy = base / "tests" / "tests_data" / "SN2025cy_lcbolinput_v2.dat"
    infile_2025ifq = base / "tests" / "tests_data" / "SN2025ifq_lcbolinput.dat"

    pbinfo_file = base / "tests" / "tests_data" / "pbinfo.dat"

    wizard = PseudoBoloWizard(infile_2025cy, pbinfo_file)
    selected_filters = wizard.run()


if __name__ == "__main__":
    main()