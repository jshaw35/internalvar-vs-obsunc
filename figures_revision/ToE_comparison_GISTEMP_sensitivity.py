"""
Compare ToE for different availability thresholds with the GISTEMP data.
"""

# %%

# Import packages
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# %%

def process_startyears(
    startyears: xr.DataArray,
    earliest_startyear: int,
    fillvalue: int,
    ens_dim: str = "realization",
):

    startyears_mid = startyears.quantile(0.75, dim=ens_dim)
    
    # Set values less than earliest_startyear to earliest_startyear.
    startyears_final = xr.where(
        startyears_mid > earliest_startyear,
        startyears_mid, 
        earliest_startyear,
    )
    # Revert masking to a nan
    startyears_final = startyears_final.where(startyears_final != fillvalue)
    
    return startyears_final
    

def to_png(file, filename, loc='/glade/u/home/jonahshaw/figures/',dpi=200,ext='png',**kwargs):
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
# Load GISTEMP data

# Processed ToE data
toe_savedir = "/glade/u/home/jonahshaw/w/trend_uncertainty/nathan/ToE_output"
gistemp_regional_toe = xr.open_dataset(f"{toe_savedir}/all_ToE.IPCCRegions.nc")
gistemp_70_95_regional_toe = xr.open_dataset(f"{toe_savedir}/all_ToE.IPCCRegions.GISTEMP_sensitivity.nc")

# Trend data
obs_tseries_dir = "/glade/work/jonahshaw/trend_uncertainty/nathan/OBS_LENS/"

gistemp_5x5_95_dir = 'GISTEMP_5x5/20240820/xagg_correctedtime/threshold_0.95/'
gistemp_5x5_90_dir = 'GISTEMP_5x5/20240820/xagg_correctedtime/'
gistemp_5x5_70_dir = 'GISTEMP_5x5/20240820/xagg_correctedtime/threshold_0.70/'

gistemp_5x5_regional_trends_70_filepath = '%s/%s/ensembleChunks_0001_0200.trends.movingstartdate.IPCCRegions.190001-202012.nc' % (obs_tseries_dir,gistemp_5x5_70_dir)
gistemp_5x5_regional_trends_90_filepath = '%s/%s/ensembleChunks_0001_0200.trends.movingstartdate.IPCCRegions.190001-202012.nc' % (obs_tseries_dir,gistemp_5x5_90_dir)
gistemp_5x5_regional_trends_95_filepath = '%s/%s/ensembleChunks_0001_0200.trends.movingstartdate.IPCCRegions.190001-202012.nc' % (obs_tseries_dir,gistemp_5x5_95_dir)

# Computed start years
gistemp_5x5_regional_startyears_70_filepath = '%s/%s/ensembleChunks_0001_0200.startyears.IPCCRegions.190001-202012.nc' % (obs_tseries_dir, gistemp_5x5_70_dir)
gistemp_5x5_regional_startyears_90_filepath = '%s/%s/ensembleChunks_0001_0200.startyears.IPCCRegions.190001-202012.nc' % (obs_tseries_dir, gistemp_5x5_90_dir)
gistemp_5x5_regional_startyears_95_filepath = '%s/%s/ensembleChunks_0001_0200.startyears.IPCCRegions.190001-202012.nc' % (obs_tseries_dir, gistemp_5x5_95_dir)

# Load the trend data
gistemp_regional_trends_70_all = xr.open_dataarray(gistemp_5x5_regional_trends_70_filepath).sel(startyear=1).drop_vars(['startyear']).chunk({"RegionIndex": 1})
gistemp_regional_trends_90_all = xr.open_dataarray(gistemp_5x5_regional_trends_90_filepath).sel(startyear=1).drop_vars(['startyear']).chunk({"RegionIndex": 1})
gistemp_regional_trends_95_all = xr.open_dataarray(gistemp_5x5_regional_trends_95_filepath).sel(startyear=1).drop_vars(['startyear']).chunk({"RegionIndex": 1})

# Load the start year data
gistemp_regional_70_startyears = xr.open_dataarray(gistemp_5x5_regional_startyears_70_filepath)
gistemp_regional_90_startyears = xr.open_dataarray(gistemp_5x5_regional_startyears_90_filepath)
gistemp_regional_95_startyears = xr.open_dataarray(gistemp_5x5_regional_startyears_95_filepath)

# %%

earliest_startyear = 1900
fillvalue = 2100

gistemp_regional_70_startyears = process_startyears(
    gistemp_regional_70_startyears,
    earliest_startyear=earliest_startyear,
    fillvalue=fillvalue,
).drop_vars("quantile").squeeze()

gistemp_regional_90_startyears = process_startyears(
    gistemp_regional_90_startyears,
    earliest_startyear=earliest_startyear,
    fillvalue=fillvalue,
).drop_vars("quantile").squeeze()

gistemp_regional_95_startyears = process_startyears(
    gistemp_regional_95_startyears,
    earliest_startyear=earliest_startyear,
    fillvalue=fillvalue,
).drop_vars("quantile").squeeze()

# %%
# PIC model data
pic_tseries_dir = '/glade/work/jonahshaw/trend_uncertainty/nathan/CMIP6_PIC/'
cesm1_dir       = 'CESM1/xagg'
mpi_dir         = 'MPI-GE/xagg'
canesm2_dir     = 'CanESM2/xagg'
esm2m_dir       = 'ESM2M/xagg'

cesm1_regional_trends_filepath   = glob.glob('%s/%s/*.1900trends.Percentiles*.nc' % (pic_tseries_dir,cesm1_dir))
mpi_regional_trends_filepath     = glob.glob('%s/%s/*.1900trends.Percentiles*.nc' % (pic_tseries_dir,mpi_dir))
canesm2_regional_trends_filepath = glob.glob('%s/%s/*.1900trends.Percentiles*.nc' % (pic_tseries_dir,canesm2_dir))
esm2m_regional_trends_filepath   = glob.glob('%s/%s/*.1900trends.Percentiles*.nc' % (pic_tseries_dir,esm2m_dir))

cesm1_pic_regional_trends_all   = xr.open_dataarray(*cesm1_regional_trends_filepath).chunk({"RegionIndex": 1})
cesm1_pic_regional_trends_all["percentile"] = cesm1_pic_regional_trends_all["percentile"] * 100

mpi_pic_regional_trends_all     = xr.open_dataarray(*mpi_regional_trends_filepath).chunk({"RegionIndex": 1})
mpi_pic_regional_trends_all["percentile"] = mpi_pic_regional_trends_all["percentile"] * 100

canesm2_pic_regional_trends_all = xr.open_dataarray(*canesm2_regional_trends_filepath).chunk({"RegionIndex": 1})
canesm2_pic_regional_trends_all["percentile"] = canesm2_pic_regional_trends_all["percentile"] * 100

esm2m_pic_regional_trends_all   = xr.open_dataarray(*esm2m_regional_trends_filepath).chunk({"RegionIndex": 1})
esm2m_pic_regional_trends_all["percentile"] = esm2m_pic_regional_trends_all["percentile"] * 100

# %%

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

# %%

region_index = 10
region_index_list = [0, 9, 10, 11, 21, 22, 24, 31]
fig, axs = plt.subplots(2, 4, figsize=(16,8.5))
fig.subplots_adjust(hspace=0.3, wspace=0.4)
axs = axs.flat

for ax, region_index in zip(axs, region_index_list):
    
    gistemp90_label = f"90%: {int(gistemp_regional_90_startyears.sel(RegionIndex=region_index).values)}"
    gistemp70_label = f"70%: {int(gistemp_regional_70_startyears.sel(RegionIndex=region_index).values)}"
    try:
        gistemp95_label = f"95%: {int(gistemp_regional_95_startyears.sel(RegionIndex=region_index).values)}"
    except:
        gistemp95_label = "95%: N/A"

    _recordlength_70 = 2021 - gistemp_regional_70_startyears.sel(RegionIndex=region_index)
    _recordlength_90 = 2021 - gistemp_regional_90_startyears.sel(RegionIndex=region_index)
    _recordlength_95 = 2021 - gistemp_regional_95_startyears.sel(RegionIndex=region_index)

    ax.plot(
        gistemp_regional_trends_70_all.sel(RegionIndex=region_index, duration=slice(None, _recordlength_70)).median("realization").duration,
        gistemp_regional_trends_70_all.sel(RegionIndex=region_index, duration=slice(None, _recordlength_70)).median("realization"),
        label=gistemp70_label,
    )

    ax.plot(
        gistemp_regional_trends_90_all.sel(RegionIndex=region_index, duration=slice(None, _recordlength_90)).median("realization").duration,
        gistemp_regional_trends_90_all.sel(RegionIndex=region_index, duration=slice(None, _recordlength_90)).median("realization"),
        label=gistemp90_label,
    )

    ax.plot(
        gistemp_regional_trends_95_all.sel(duration=slice(None, _recordlength_95)).duration,
        gistemp_regional_trends_95_all.sel(RegionIndex=region_index, duration=slice(None, _recordlength_95)).median("realization"),
        label=gistemp95_label,
    )

    ax.fill_between(
        cesm1_pic_regional_trends_all.sel(RegionIndex=region_index,).duration,
        cesm1_pic_regional_trends_all.sel(RegionIndex=region_index, percentile=2.5),
        cesm1_pic_regional_trends_all.sel(RegionIndex=region_index, percentile=97.5),
        color="grey",
        # label="PIC",
        alpha=0.5
    )
    ax.set_xlim(0, 120)
    ax.set_xlabel("Trend Duration (years)")
    ax.set_ylabel("Trend (°C/decade)")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
    ax.set_ylim(-0.1, 0.1)
    ax.set_title(f"{cesm1_pic_regional_trends_all.sel(RegionIndex=region_index,).RegionName.values}")

    ax.legend()

to_png(
    fig,
    f"GISTEMP_sensitivity_ToE_comparison_coneplots",
    dpi=200,
    ext="pdf",
    bbox_inches="tight",
)

# %%

regions_list = [0, 9, 10, 11, 21, 22, 24, 31]

# Combine ToE data
toe_90 = gistemp_regional_toe["ToE"].sel(obs="GISTEMP_5x5", model="CESM1", RegionIndex=regions_list)
toe_sensitivity = gistemp_70_95_regional_toe["ToE"].sel(obs=["GISTEMP_5x5_70", "GISTEMP_5x5_95"], model="CESM1", RegionIndex=regions_list)

toe_gistemp_all = xr.concat([toe_90.assign_coords({"obs":"GISTEMP_5x5_90"}).expand_dims("obs"), toe_sensitivity], dim="obs")
toe_gistemp_strict_all = toe_gistemp_all.where(~np.isnan(toe_gistemp_all), np.inf).quantile(q=0.95, dim="realization")
toe_gistemp_strict_all = toe_gistemp_strict_all.where(~np.isinf(toe_gistemp_strict_all), np.nan).drop("quantile")

toe_gistemp_median_all = toe_gistemp_all.where(~np.isnan(toe_gistemp_all), np.inf).median(dim="realization")



# %%

# Create a scatter plot similar to the comparison of IPCC regions to show the ToE for different availability thresholds
# This is awesome, we basically see that the uncertainty due to our threshold is: 
# 1. Only in the GISTEMP data product
# 2. Only in certain regions
# 3. In regions that already have high ToE uncertainty that has been sampled.
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

for i, region_index in enumerate(regions_list):
    _data_strict = toe_gistemp_strict_all.sel(RegionIndex=region_index)
    _data_median = toe_gistemp_median_all.sel(RegionIndex=region_index)
    
    offset = 0.17
    
    ax.scatter(
        i + offset,
        _data_strict.sel(obs="GISTEMP_5x5_70"),
        label=f"70% threshold",
        color="blue",
        marker="x",
        s=150,
    )
    ax.scatter(
        i + offset,
        _data_strict.sel(obs="GISTEMP_5x5_90"),
        label=f"90% threshold",
        color="green",
        marker="+",
        s=150,
    )
    ax.scatter(
        i + offset,
        _data_strict.sel(obs="GISTEMP_5x5_95"),
        label=f"95% threshold",
        color="red",
        marker="$\\bigtriangleup$",
        s=200,
    )

    ax.scatter(
        i - offset,
        _data_median.sel(obs="GISTEMP_5x5_70"),
        label=f"70% threshold",
        color="blue",
        marker="x",
        s=150,
    )
    ax.scatter(
        i - offset,
        _data_median.sel(obs="GISTEMP_5x5_90"),
        label=f"90% threshold",
        color="green",
        marker="+",
        s=150,
    )
    ax.scatter(
        i - offset,
        _data_median.sel(obs="GISTEMP_5x5_95"),
        label=f"95% threshold",
        color="red",
        marker="$\\bigtriangleup$",
        # markeredgewidth=2,
        s=200,
    )

ax.vlines(
    x=np.arange(0, len(regions_list)),
    ymin=1900,
    ymax=2021,
    color="black",
    linestyle="--",
    linewidth=0.5,
)
ax.set_ylim(1920, 2021)
ax.set_xticks(np.arange(0, len(regions_list)))
ax.set_xticklabels(toe_gistemp_all.sel(RegionIndex=regions_list).RegionName.values, rotation=45, fontsize=13)

ax.set_ylabel("Year of Emergence", fontsize=13)
ax.set_xlabel("Region", fontsize=13)

handles, labels = ax.get_legend_handles_labels()
plt.legend(handles[:3], labels[:3])#, loc="upper left")

to_png(
    fig,
    f"GISTEMP_sensitivity_ToE_comparison_scatterplot",
    dpi=200,
    ext="pdf",
    bbox_inches="tight",
)

# %%
