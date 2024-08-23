"""
Script to download ESM-2M tas data from the ESGF.

"""

import glob
import os
from subprocess import run
import numpy as np

if __name__ == "__main__":

    # The ESM-2M data is available in 5-year increments, which means there are 100 files.

    filepath_base = "https://esgf.ceda.ac.uk/thredds/dodsC/esg_dataroot/cmip5/output1/NOAA-GFDL/GFDL-ESM2M/piControl/mon/atmos/Amon/r1i1p1/v20130214/tas/tas_Amon_GFDL-ESM2M_piControl_r1i1p1_"
    run_path = "/glade/derecho/scratch/jonahshaw/CMIP5/GFDL-ESM2M/" # "/glade/u/home/jonahshaw"

    for startyear in np.arange(1, 5, 500):

        stopyear = startyear + 4

        filepath = f"{filepath_base}{startyear:04}01-{stopyear:04}12.nc"
        print(filepath)

        wget_cmd = \
            f"wget {filepath}"
        print(wget_cmd)
            
        run(wget_cmd, cwd=run_path, shell=True)
        break
    
# https://esgf.ceda.ac.uk/thredds/dodsC/esg_dataroot/cmip5/output1/NOAA-GFDL/GFDL-ESM2M/piControl/mon/atmos/Amon/r1i1p1/v20130214/tas/tas_Amon_GFDL-ESM2M_piControl_r1i1p1_000101-000512.nc
# 049601-050012

    # for _var in all_h0_variables:
    #     outfile = f"{h0a_files[0].split('/')[-1][:-10]}{_var}.{_year}.nc"
    #     if os.path.exists(os.path.join(save_dir, outfile)):  # Do not repeat existing files
    #         continue
    #     subset_files_cmd = \
    #         f"ncrcat " \
    #         f"-v {_var} " \
    #         f"{cesmcase_file_str}" \
    #         f"{os.path.join(save_dir, outfile)}"

    #     print(subset_files_cmd)

    #     run(subset_files_cmd, cwd=run_path, shell=True)