import pathlib 
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import glob
import os
import math
import scienceplots
matplotlib.style.use(['science'])

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pseudobolo.aux import rd_lcbol_data

# ==========================================
# CONFIGURATION
# ==========================================
#Define source folder for IDL files:
IDL_DIR = ROOT / "tests" / "tests_data"
PYTHON_DIR = ROOT / "tests" / "tests_results"
IDL_FILTERS_DIR = IDL_DIR/ "filter_interp"
PYTHON_FILTERS_DIR = PYTHON_DIR / "filter_interp"


# List of filters to look for (in order of plotting)
# You can add more like 'u', 'g', 'r', 'i', 'z' if needed
FILTERS_TO_FIND = ['U', 'B', 'V', 'R', 'I', 'J', 'H', 'Ks', 'g']

# Figure settings
FIG_SIZE = (15, 12)  # Width, Height in inches

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def find_file_with_filter(folder, filter_name):
    """
    Finds a file in a folder that contains '_filtername_' or similar.
    Adjust the pattern matching inside the glob if your naming is unique.
    """
    # Look for files containing the filter name (case insensitive usually safer to control)
    # This pattern looks for *B* inside the filename.
    # We look for _B_ or _B. to avoid finding 'B' inside 'LCBOL'.
    search_patterns = [
        f"*{filter_name}_interp.dat",   # Common IDL pattern
        f"*{filter_name}_py.dat",       # Common Python pattern
        f"*_{filter_name}_*.dat"       # Catch-all
    ]
    
    for pattern in search_patterns:
        files = glob.glob(os.path.join(folder, pattern))
        if files:
            return files[0] # Return the first match
    return None





def main():
    #Load IDL results:
    data_idl = IDL_DIR / "sn2002bo_lcbol_UBVRI.dat"
    t_idl, Lbol_idl, Lbolerr_idl = np.loadtxt(
        data_idl,
        comments="#",
        unpack=True,
        usecols=(0, 1, 2)
    )

    #Load Python results:
    data_py = PYTHON_DIR / "sn2002bo_lcbol_UBVRI_py.dat"
    t_py, Lbol_py, Lbolerr_py = np.loadtxt(
        data_py,
        comments="#",
        unpack=True,
        usecols=(0, 1, 2)
    )
    
    flux_idl = IDL_DIR / "sn2002bo_lcbol_UBVRI_flux_int.dat"
    t_idl, F_idl, Ferr_idl = np.loadtxt(
        flux_idl,
        comments="#",
        unpack=True,
        usecols=(0, 1, 2)
    )
    
    flux_py = PYTHON_DIR / "sn2002bo_lcbol_UBVRI_flux_int_py.dat"
    t_py, F_py, Ferr_py = np.loadtxt(
        flux_py,
        comments="#",
        unpack=True,
        usecols=(0, 1, 2)
    )

    #Compare the two results:
    fig1, axes1 = plt.subplots(2, 1, figsize=(8, 10))
    fig1.suptitle('Comparison of IDL and Python Bolometric Light Curve Results', fontsize=16)
    axes1[0].errorbar(t_idl, Lbol_idl, yerr=Lbolerr_idl, fmt='o', label='IDL', alpha=0.7)
    axes1[0].errorbar(t_py, Lbol_py, yerr=Lbolerr_py, fmt='s', label='Python', alpha=0.7)
    axes1[0].set_xlabel('Time (days)')
    axes1[0].set_ylabel('Bolometric Luminosity (erg/s)')
    axes1[0].set_title('Bolometric Light Curve Comparison')
    axes1[0].legend()
    
    axes1[1].errorbar(t_idl, F_idl, yerr=Ferr_idl, fmt='o', label='IDL', alpha=0.7)
    axes1[1].errorbar(t_py, F_py, yerr=Ferr_py, fmt='s', label='Python', alpha=0.7)
    axes1[1].set_xlabel('Time (days)')
    axes1[1].set_ylabel('Integrated Flux (erg/s/cm²)')
    axes1[1].set_title('Integrated Flux Comparison')
    axes1[1].legend()
    
    
    fig1.savefig(PYTHON_DIR / "idl_python_comparison_UBVRI.png", dpi=300)
    
    # ==========================================
    # PLOTTING LOGIC
    # ==========================================

    # Create 3x3 Grid
    fig, axes = plt.subplots(3, 3, figsize=FIG_SIZE)
    axes = axes.flatten() # Flatten 2D grid to 1D list for easy looping
    fig.subplots_adjust(
            left=0.06,
            right=0.97,
            top=0.88,
            bottom=0.06,   # <- keeps plots away from buttons
            hspace=1.0,   # <- bigger vertical space between rows of subplots
            wspace=0.30,
        )

    found_count = 0

    for i, filt in enumerate(FILTERS_TO_FIND):
        if i >= 9: break # Safety break if list is too long for 3x3
        
        ax = axes[i]
        
        # 1. Find Files
        idl_file = find_file_with_filter(IDL_FILTERS_DIR, filt)
        py_file  = find_file_with_filter(PYTHON_FILTERS_DIR, filt)
        
        has_data = False
        # 2. Plot IDL Data
        if idl_file:
            lcdata = pd.read_csv(idl_file, sep=r'\s+', engine='python', comment='#', 
                         header=None, names=['mjd', 'mag', 'err'])
            
            magerr_idl = lcdata.err
            mag_idl = lcdata.mag
            mjd_idl = lcdata.mjd
            
            if lcdata is not None:
                ax.errorbar(mjd_idl, mag_idl, yerr=magerr_idl, 
                            fmt='o', label='IDL', 
                            markersize=5, alpha=0.7)
                has_data = True

        # 3. Plot Python Data
        if py_file:
            _, pydata = rd_lcbol_data(py_file)
            for lc in pydata:
                print(f"Filter: {lc.filt}")
                magerr_py = lc.magerr
                mag_py = lc.mag
                mjd_py = lc.time
    
            if pydata is not None:
                # Using a generic 'x' marker for python to distinguish overlap
                ax.errorbar(mjd_py, mag_py, yerr=magerr_py,
                            fmt='o', label='Python', 
                            markersize=6, alpha=0.5)
                has_data = True
                
        # 4. Styling the Axis
        if has_data:
            ax.set_title(f"Filter: {filt}", fontweight='bold')
            ax.invert_yaxis() # Astronomy standard: brighter is up (lower mag)
            ax.set_xlabel("MJD")
            ax.set_ylabel("Magnitude")
            ax.legend()
            
            # Only add legend to the first plot to save space, or all if you prefer
            if i == 0:
                ax.legend()
            found_count += 1
        else:
            # Hide empty plots if file not found
            ax.set_visible(False)

    # Clean up layout
    plt.tight_layout()
    plt.suptitle("Comparison: IDL vs Python Interpolation", fontsize=16, y=1.02)
    plt.show()
    fig.savefig(PYTHON_DIR/"filter_by_filter_comp.png", dpi = 400)

    if found_count == 0:
        print("WARNING: No files were found. Check your folder paths and 'FILTERS_TO_FIND' list.")

    
    
    
if __name__ == "__main__":
    main()