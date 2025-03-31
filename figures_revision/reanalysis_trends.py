"""
The script creates a figure that compares trends in standard reanalysis products to the data products used in the study.
I think it is ok to do this regionally for the supplement.
"""

# %%
# Import packages
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import regionmask
import numpy as np
import os
import glob

# Import cartopy for map plotting
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# %%

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

# Load regional time series data
obslens_tseries_dir = '/glade/u/home/jonahshaw/w/trend_uncertainty/nathan/OBS_LENS/'

#### Collect GISTEMP 5x5 file paths.
gistemp_5x5_files = glob.glob('%s/GISTEMP_5x5/20240820/xagg_correctedtime/ensembleChunk_5x5_????.nc' % obslens_tseries_dir)
gistemp_5x5_files.sort()

#### Collect HadCRUT5 file paths.
hadcrut5_files = glob.glob('%s/HadCRUT5/20240820/xagg/HadCRUT.5.0.2.0.analysis.anomalies.*.nc' % obslens_tseries_dir)
hadcrut5_files = [i for i in hadcrut5_files if "Trends" not in i]
hadcrut5_files.sort()

#### Collect BEST file paths.
best_files = ['%s/BEST/20250320/xagg/Land_and_Ocean_LatLong1.nc' % obslens_tseries_dir]

# Reanalysis data
era5_files = [f"{obslens_tseries_dir}ERA5/20250323/xagg/era5.t2m.194001-202412.nc"]
merra2_files = [f"{obslens_tseries_dir}MERRA2/20250323/xagg/T2M_MERRA2_asm_mon_198001_202012.nc"]

# load the data
gistemp_ens_ds = xr.open_mfdataset(gistemp_5x5_files,combine='by_coords', chunks={"realization": 1})
hadcrut5_ens_ds = xr.open_mfdataset(hadcrut5_files,combine='by_coords', chunks={"realization": 1})
best_ds = xr.open_mfdataset(best_files,combine='by_coords').assign_coords({"source_id":"BEST"}).expand_dims("source_id")

era5_ds = xr.open_mfdataset(era5_files,combine='by_coords').assign_coords({"source_id":"ERA5"}).expand_dims("source_id")
merra2_ds = xr.open_mfdataset(merra2_files,combine='by_coords').assign_coords({"source_id":"MERRA2"}).expand_dims("source_id")

# Reduce the ensembles to their median values for comparison
gistemp_ds = gistemp_ens_ds.median('realization').compute().assign_coords({"source_id":"GISTEMP_MEDIAN"}).expand_dims("source_id")
hadcrut5_ds = hadcrut5_ens_ds.median('realization').compute().assign_coords({"source_id":"HadCRUT_MEDIAN"}).expand_dims("source_id")

# %%

data_list = [gistemp_ds, hadcrut5_ds, best_ds, era5_ds, merra2_ds]
var_list = ["tas", "tas", "temperature", "t2m", "T2M"]
source_list = ["GISTEMP", "HadCRUT", "BEST", "ERA5", "MERRA2"]
combine_list = []
time_coord = merra2_ds["time"]
for _data, _source, _tas_var in zip(data_list, source_list, var_list):
    _data = _data.sel(time=slice("1980", "2020"))
    _data['time'] = time_coord
    _data = _data.rename_vars({_tas_var:"tas"})
    # _data = _data.assign_coords({"source_id":_source})
    # _data['time'] = _data.indexes['time'].to_datetimeindex()
    combine_list.append(_data)

all_sources_ds = xr.combine_by_coords(combine_list)
all_sources_annual_ds = all_sources_ds.groupby('time.year').mean('time')
# all_sources_annual_ds = all_sources_ds.resample(time='1Y').mean()

# %%

all_sources_trend = all_sources_annual_ds.polyfit(dim='year', deg=1)
trends_ds = all_sources_trend["tas_polyfit_coefficients"].sel(degree=1).drop("degree").compute()

# %%

# send data to lon-lat space
region_mask = regionmask.defined_regions.ar6.land
all_sources_gridded_ds = project_IPCCregions_tolatlon(trends_ds, region_mask)

# %%

# Create a figure with 5 panels for the different sources
fig, axs = plt.subplots(
    3, 2,
    figsize=(12, 11), 
    subplot_kw={'projection': ccrs.PlateCarree()},
    # constrained_layout=True,
)
# fig.subplots_adjust(hspace=0.3, wspace=0.1)

# Flatten the axes for easier indexing
axs = axs.flat

# Calculate min and max for consistent colorbar
vmin = float(all_sources_gridded_ds.min())
vmax = float(all_sources_gridded_ds.max())
vext = max(abs(vmin), abs(vmax))  # For symmetric colorbar around zero

# Plot each source
sources = ["GISTEMP_MEDIAN", "HadCRUT_MEDIAN", "BEST", "ERA5", "MERRA2"]
source_labels = ["GISTEMP (median)", "HadCRUT (median)", "BEST", "ERA5", "MERRA2"]
subplot_labels = ["a", "b", "c", "d", "e"]

for i, source_id in enumerate(sources):
    ax = axs[i]
    
    # Get data for this source
    da = all_sources_gridded_ds.sel(source_id=source_id)
    
    # Plot the data
    # im = da.plot(ax=ax, cmap='RdBu_r', vmin=-vext, vmax=vext, 
    #            add_colorbar=False)
    im = ax.pcolormesh(
        da.lon,
        da.lat,
        da,
        # ax=ax,
        cmap='RdBu_r',
        vmin=-vext,
        vmax=vext,
        transform=ccrs.PlateCarree(),
        # add_colorbar=False,
    )
    # break
    # continue

    # Add geographic features
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    # ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
    
    # Set title
    ax.set_title(source_labels[i])
    
    ax.text(-0.07, 0.90, f"{subplot_labels[i]}.", transform=ax.transAxes, size=16)

# Remove any unused panel
if len(all_sources_gridded_ds.source_id) < len(axs):
    fig.delaxes(axs[5])

# Add a colorbar
cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.015])
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Temperature Trend (K/year)')

fig.suptitle('Temperature Trends 1980-2020 by Data Source', fontsize=16, y=0.93)

 # %%
# Save the figure
to_png(fig, 'fig_revision_datasource_trendcomparison', dpi=100, ext="pdf", bbox_inches='tight')

# %%
to_png(fig, 'fig_revision_datasource_trendcomparison', dpi=200, ext="png", bbox_inches='tight')

# %%
