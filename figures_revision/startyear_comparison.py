"""
Compare the region start years of the different surface temperature datasets with different availability thresholds.
"""

# %%

# Import packages
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import regionmask
import numpy as np
import os

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# %%

def get_final_startyear(
    ds: xr.Dataset,
    earliest_startyear: int=1900,
) -> xr.DataArray:
    
    """
    Calculate the final start year for analysis, taking the 75th percentile of available
    start years across realizations, but enforcing a minimum earliest start year.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Dataset containing a 'startyear' variable with a 'realization' dimension.
    earliest_startyear : int or float
        The earliest allowable start year. If the 75th percentile of startyears is 
        earlier than this value, this value will be used instead.

    Returns
    -------
    xr.DataArray
        The final start year, which is the maximum of the 75th percentile of startyears
        and the earliest_startyear parameter.
    """

    startyears = ds["year"]
    startyear_mid = startyears.quantile(0.75, dim="realization")

    startyear_final = xr.where(
        startyear_mid > earliest_startyear,
        startyear_mid, 
        earliest_startyear,
    )
    return startyear_final


def project_IPCCregions_tolatlon(
    data: xr.DataArray,
    region_mask: regionmask.Regions=regionmask.defined_regions.ar6.land,
    offset: int=-1000,
) -> xr.DataArray:
    """
    Project the IPCC regions to lat-lon coordinates for plotting.
    """

    unc_mask = region_mask.mask(np.arange(-179.5, 180, 0.5),np.arange(-89.5, 90, 0.5),)

    _startyear_mask   = unc_mask.copy()

    del unc_mask

    offset = -1000 # Need an offset or it may confuse regions with the delays (delays are on [0,40] similar to the region indices)
    for i in region_mask.numbers:
        
        _startyear_mask = xr.where(_startyear_mask==i, data.sel(RegionIndex=i) + offset, _startyear_mask)
    
    return _startyear_mask - offset


def to_png(
    file,
    filename,
    loc='/glade/u/home/jonahshaw/figures/',
    dpi=200,
    ext='png',
    **kwargs,
):
    '''
    Simple function for one-line saving.
    Saves to "/glade/u/home/jonahshaw/figures" by default
    '''
    output_dir = loc
    full_path = '%s%s.%s' % (output_dir,filename,ext)

    if not os.path.exists(output_dir + filename):
        file.savefig(full_path,format=ext, dpi=dpi,**kwargs)
        
    else:
        print('File already exists, rename or delete.')


# %%

# Load the data
obslens_tseries_dir = '/glade/u/home/jonahshaw/w/trend_uncertainty/nathan/OBS_LENS/'

gistemp_90_datapath = f"{obslens_tseries_dir}GISTEMP_5x5/20240820/xagg_correctedtime/ensembleChunks_0001_0200.startyears.IPCCRegions.190001-202012.nc"
gistemp_95_datapath = f"{obslens_tseries_dir}GISTEMP_5x5/20240820/xagg_correctedtime/threshold_0.95/ensembleChunks_0001_0200.startyears.IPCCRegions.190001-202012.nc"
gistemp_70_datapath = f"{obslens_tseries_dir}GISTEMP_5x5/20240820/xagg_correctedtime/threshold_0.70/ensembleChunks_0001_0200.startyears.IPCCRegions.190001-202012.nc"

hadcrut_90_datapath = f"{obslens_tseries_dir}HadCRUT5/20240820/xagg/HadCRUT.5.0.2.0.001_0200.startyears.IPCCRegions.190001-202412.nc"
hadcrut_95_datapath = f"{obslens_tseries_dir}HadCRUT5/20240820/xagg/threshold_0.95/HadCRUT.5.0.2.0.001_0200.startyears.IPCCRegions.190001-202412.nc"
hadcrut_70_datapath = f"{obslens_tseries_dir}HadCRUT5/20240820/xagg/threshold_0.70/HadCRUT.5.0.2.0.001_0200.startyears.IPCCRegions.190001-202412.nc"

best_90_datapath = f"{obslens_tseries_dir}BEST/20250320/xagg/Land_and_Ocean_LatLong1.startyears.IPCCRegions.190001-202012.nc"
best_95_datapath = f"{obslens_tseries_dir}BEST/20250320/xagg/threshold_0.95/Land_and_Ocean_LatLong1.startyears.IPCCRegions.190001-202012.nc"
best_70_datapath = f"{obslens_tseries_dir}BEST/20250320/xagg/threshold_0.70/Land_and_Ocean_LatLong1.startyears.IPCCRegions.190001-202012.nc"

gistemp_90_ds = xr.open_dataset(gistemp_90_datapath)
gistemp_90_ds = get_final_startyear(gistemp_90_ds).drop_vars("quantile").assign_coords({"source_id":"GISTEMP","threshold":0.90}).expand_dims(["source_id","threshold"])

gistemp_95_ds = xr.open_dataset(gistemp_95_datapath)
gistemp_95_ds = get_final_startyear(gistemp_95_ds).drop_vars("quantile").assign_coords({"source_id":"GISTEMP","threshold":0.95}).expand_dims(["source_id","threshold"])

gistemp_70_ds = xr.open_dataset(gistemp_70_datapath)
gistemp_70_ds = get_final_startyear(gistemp_70_ds).drop_vars("quantile").assign_coords({"source_id":"GISTEMP","threshold":0.70}).expand_dims(["source_id","threshold"])

hadcrut_90_ds = xr.open_dataset(hadcrut_90_datapath)
hadcrut_90_ds = get_final_startyear(hadcrut_90_ds).drop_vars("quantile").assign_coords({"source_id":"HadCRUT","threshold":0.90}).expand_dims(["source_id","threshold"])

hadcrut_95_ds = xr.open_dataset(hadcrut_95_datapath)
hadcrut_95_ds = get_final_startyear(hadcrut_95_ds).drop_vars("quantile").assign_coords({"source_id":"HadCRUT","threshold":0.95}).expand_dims(["source_id","threshold"])

hadcrut_70_ds = xr.open_dataset(hadcrut_70_datapath)
hadcrut_70_ds = get_final_startyear(hadcrut_70_ds).drop_vars("quantile").assign_coords({"source_id":"HadCRUT","threshold":0.70}).expand_dims(["source_id","threshold"])

best_90_ds = xr.open_dataset(best_90_datapath).squeeze().assign_coords({"source_id":"BEST","threshold":0.90}).expand_dims(["source_id","threshold"])["year"]
best_95_ds = xr.open_dataset(best_95_datapath).squeeze().assign_coords({"source_id":"BEST","threshold":0.95}).expand_dims(["source_id","threshold"])["year"]
best_70_ds = xr.open_dataset(best_70_datapath).squeeze().assign_coords({"source_id":"BEST","threshold":0.70}).expand_dims(["source_id","threshold"])["year"]

all_startyears_ds = xr.combine_by_coords(
    [
        gistemp_90_ds, 
        gistemp_95_ds, 
        gistemp_70_ds, 
        hadcrut_90_ds, 
        hadcrut_95_ds, 
        hadcrut_70_ds, 
        best_90_ds, 
        best_95_ds, 
        best_70_ds,
    ]
)

# %%

# send data to lon-lat space
region_mask = regionmask.defined_regions.ar6.land
all_startyears_gridded_ds = project_IPCCregions_tolatlon(all_startyears_ds,region_mask)

# %%
# Create a 3x3 panel plot of the start years for different regions, datasets, and availability thresholds
# Only GISTEMP shows much change in start years with different availability thresholds. Regions with change are:
# 1. Northern South America (10)
# 2. North-Western South America (9)
# 3. North-Eastern South America (11)
# 4. Greenland/Iceland (0)
# 5. Western Africa (21)
# 6. Central Africa (22)
# 7. South Eastern Africa (24)
# 8. Russian Far East (31)
# Indices are 0-based: [0, 9, 10, 11, 21, 22, 24, 31]

# Setup figure and colormap
fig = plt.figure(figsize=(15, 10))
_cmap = plt.cm.viridis_r
_norm = plt.Normalize(1900, 1980)  # Adjust based on your data range

# Define datasets and thresholds for iteration
datasets = ["GISTEMP", "HadCRUT", "BEST"]
thresholds = [0.70, 0.90, 0.95]

# Create 3x3 grid of subplots
for i, dataset in enumerate(datasets):
    for j, threshold in enumerate(thresholds):
        ax_idx = i * 3 + j + 1
        _ax = plt.subplot(3, 3, ax_idx, projection=ccrs.Robinson())

        # Extract data for this dataset and threshold
        _mask = all_startyears_gridded_ds["year"].sel(source_id=dataset, threshold=threshold)

        # Plot the data
        im = _ax.pcolormesh(
            _mask.lon, _mask.lat, _mask,
            transform=ccrs.PlateCarree(),
            norm=_norm,
            cmap=_cmap,
        )

        _ax.coastlines()
        _ax.set_global()
        _label = f"{dataset}, threshold={threshold}"
        _ax.set_title(_label, fontsize=16)

# Add colorbar
cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])  # [left, bottom, width, height]
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('Start Year', fontsize=14)

plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to make room for colorbar
plt.suptitle('Start Years by Dataset and Availability Threshold', fontsize=20, y=0.95)

to_png(fig,'startyear_comparison', dpi=200, bbox_inches='tight', ext="pdf")

plt.show()

# %%
