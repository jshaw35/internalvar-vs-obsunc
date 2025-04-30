"""
Create better example figure for the GISTEMP ensemble.

"""

# %%
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import warnings
 
# %%

obslens_tseries_dir = '/glade/u/home/jonahshaw/w/trend_uncertainty/nathan/OBS_LENS/'


# __Define functions:__
def to_png(file, filename, loc='/glade/u/home/jonahshaw/figures/',dpi=200,ext='png',**kwargs):
    '''
    Simple function for one-line saving.
    Saves to "/glade/u/home/jonahshaw/figures" by default
    '''
    output_dir = loc
    full_path = '%s%s.%s' % (output_dir,filename,ext)

    if not os.path.exists(output_dir + filename):
        file.savefig(full_path,format=ext, dpi=dpi,**kwargs)
#         file.clf()
        
    else:
        print('File already exists, rename or delete.')


def get_trends_allstartyears(
    data,
    duration,
    startyears,
    dim='year',
    dask=False,
):
    '''
    Calculate: 
    a. trends of a given duration
    '''
    
    # Initialize list to save to
    trends_list = []
    
    for i,startyear in enumerate(startyears):
        
        _startyr = startyear
        _endyr   = startyear + duration - 1

        _tsel = data.sel({dim:slice(_startyr,_endyr)}) # index differently here because the dates are different

        # Calculate the slope
        # print(_tsel)
        # print(dim)
        if dask:
            _tsel_polyfit = dask.delayed(xr.DataArray.polyfit)(_tsel, dim=dim, deg=1, skipna=True)['polyfit_coefficients'].sel(degree=1)
        else:
            _tsel_polyfit = xr.DataArray.polyfit(_tsel, dim=dim, deg=1, skipna=True)['polyfit_coefficients'].sel(degree=1)
        _tsel_slopes  = _tsel_polyfit.drop_vars('degree')
        
        trends_list.append(_tsel_slopes.assign_coords({'startyear':startyear}).expand_dims('startyear'))
    
    if dask:
        trends_list = dask.compute(*trends_list)
    out = xr.concat(trends_list, dim='startyear').assign_coords({'duration': duration}).expand_dims('duration')
        
    return out


def compute_trends_wrapper(
    data,
    durations=np.arange(5,81,),
    time_dim="year",
    metadata=True,
    mask_nan=False,
    dask=False,
    startyears=None,
    **kwargs,
):
    '''
    Wrapper for running 'get_allvar_allstartyears' for different durations.
    Setup with correct startyears and concatenate at the end.
    '''
    
    first_year = data[time_dim][0]
    last_year  = data[time_dim][-1]

    trends_allstartyear_allduration_list = []

    for duration in durations:
        print(duration,end=' ')
        if startyears is None:
            _startyears = np.arange(first_year,last_year+2-duration,1)
        else:
            _startyears = startyears

        if dask:
            allvar_onedur_ds = dask.delayed(get_trends_allstartyears)(
                data,
                duration=duration,
                startyears=_startyears,
                dim=time_dim,
                **kwargs
            )
        else:
            allvar_onedur_ds = get_trends_allstartyears(
                data,
                duration=duration,
                startyears=_startyears,
                dim=time_dim,
                **kwargs
            )            
        
        trends_allstartyear_allduration_list.append(allvar_onedur_ds)
    
    if dask:
        trends_allstartyear_allduration_list = dask.compute(*trends_allstartyear_allduration_list)
    trends_allstartyear_allduration_ds = xr.concat(trends_allstartyear_allduration_list, dim='duration')
    del trends_allstartyear_allduration_list

    # Add metadata
    if metadata:
        trends_allstartyear_allduration_ds = trends_allstartyear_allduration_ds.assign_coords({"RegionName":   ("RegionIndex", data.RegionName.data)})
        trends_allstartyear_allduration_ds = trends_allstartyear_allduration_ds.assign_coords({"RegionAbbrev": ("RegionIndex", data.RegionAbbrev.data)})

    trends_allstartyear_allduration_ds.name = 'TAS_trends'
    
    # Mask where the data is a nan at the end of the time series.
    if mask_nan:
        trends_allstartyear_allduration_ds = trends_allstartyear_allduration_ds.where(~np.isnan(data.rename({time_dim:"duration"})))
    
    return trends_allstartyear_allduration_ds


# %%
### Collect file paths.
#### Collect GISTEMP 5x5 file paths.
gistemp_5x5_files = glob.glob('%s/GISTEMP_5x5/20240820/xagg_correctedtime/ensembleChunks*.reindexed.IPCCRegions.*.nc' % obslens_tseries_dir)
gistemp_5x5_files.sort()

gistemp_tas_var = 'tas'
gistemp_5x5_ds = xr.open_dataset(*gistemp_5x5_files)

warnings.filterwarnings("ignore", r"Polyfit may be poorly conditioned")
warnings.filterwarnings("ignore", r"__array__ implementation doesn't accept a copy keyword, so passing copy=False failed. __array__ must implement 'dtype' and 'copy' keyword arguments.")

gistemp_trends_all = xr.open_dataset('%s/%s/ensembleChunks_0001_0200.trends.movingstartdate.IPCCRegions.190001-202012.nc' % (obslens_tseries_dir,'GISTEMP_5x5/20240820/xagg_correctedtime/'))

# Load startyear data
obs_tseries_dir = '/glade/work/jonahshaw/trend_uncertainty/nathan/OBS_LENS/'
gistemp_5x5_dir = 'GISTEMP_5x5/20240820/xagg_correctedtime/'

gistemp_regional_startyears_filepath  = glob.glob('%s/%s/*startyears.IPCCRegions*.nc' % (obs_tseries_dir,gistemp_5x5_dir))
gistemp_regional_startyears = xr.open_dataset(*gistemp_regional_startyears_filepath)

# %%
# Load processed model data (unforced trend percentiles)

# I need to go from the CAM output variables to the CMOR/CMIP standard names.
pic_tseries_dir = '/glade/work/jonahshaw/trend_uncertainty/nathan/CMIP6_PIC/'
cesm1_dir       = 'CESM1/xagg/'

cesm1_cmor_var_dict = {'ts':'TS','tas':'TREFHT','psl':'PSL'}

cesm1_regional_trends_filepath   = glob.glob('%s/%s/*.1900trends.Percentiles*.nc' % (pic_tseries_dir,cesm1_dir))
cesm1_pic_regional_trends_all   = xr.open_dataarray(*cesm1_regional_trends_filepath)



# %%

region_index = 34

obs_data = gistemp_5x5_ds[gistemp_tas_var].sel(RegionIndex=region_index)
obs_data = obs_data.rename({'recordlength':'year'})
obs_data = obs_data.assign_coords({'year':obs_data.year + 1900 - 1})

model_data = cesm1_pic_regional_trends_all.sel(RegionIndex=region_index)
model_data = model_data.rename({'duration':'year'})
model_data = model_data.assign_coords({'year':model_data.year + 1900 - 1})


region_name = obs_data.RegionName


fig, axs = plt.subplots(1, 3, figsize=(30,8))
axs = axs.flat

plt.subplots_adjust(wspace=0.25)

# Plot the "deterministic" ensemble mean
axs[0].plot(
    obs_data.year,
    obs_data.median(dim='realization'),
    color='blue',
    alpha=1,
    label='GISTEMP',
)

for i in obs_data.realization:
    axs[1].plot(
        obs_data.year,
        obs_data.sel(realization=i),
        color='red',
        linewidth=0.5,
        alpha=0.1,
    )

# Plot both
axs[2].plot(
    obs_data.year,
    obs_data.median(dim='realization'),
    color='blue',
    alpha=1,
    label='GISTEMP',
    zorder=1000,
)

for i in obs_data.realization:
    axs[2].plot(
        obs_data.year,
        obs_data.sel(realization=i),
        color='red',
        linewidth=0.5,
        alpha=0.1,
    )

# Adjust axes and labels
# even axes:
for _ax in axs:
    _ax.set_xlim(1900,2020)
    _ax.set_ylim(-1.6, 2)

    _ax.tick_params(axis='both', labelsize=18)

    _ax.set_ylabel('Surface Temperature Anomaly (K)',fontsize=18)
    _ax.set_xlabel('Year',fontsize=18)

to_png(file=fig,filename='thesis_methods1c',
       dpi=300,
       ext='pdf',
       bbox_inches='tight',
)

# %%

fig, axs = plt.subplots(1, 2, figsize=(20, 8))
axs = axs.flat
plt.subplots_adjust(wspace=0.25)

# Time series data
obs_data = gistemp_5x5_ds[gistemp_tas_var].sel(RegionIndex=region_index)
obs_data = obs_data.rename({'recordlength':'year'})
obs_data = obs_data.assign_coords({'year':obs_data.year + 1900 - 1})

model_data = cesm1_pic_regional_trends_all.sel(RegionIndex=region_index)
model_data = model_data.rename({'duration':'year'})
model_data = model_data.assign_coords({'year':model_data.year + 1900 - 1})


ax = axs[0]

# Plot the "deterministic" ensemble mean
ax.plot(
    obs_data.year,
    obs_data.median(dim='realization'),
    color='blue',
    alpha=1,
    label='GISTEMP',
    zorder=1000,
)

for i in obs_data.realization:
    ax.plot(
        obs_data.year,
        obs_data.sel(realization=i),
        color='red',
        linewidth=0.5,
        alpha=0.1,
    )

# Adjust axes and labels
# even axes:
ax.set_xlim(1900,2020)
ax.set_ylim(-1.6, 2)

ax.tick_params(axis='both', labelsize=18)

ax.set_ylabel('Surface Temperature Anomaly (K)',fontsize=18)
ax.set_xlabel('Year',fontsize=18)

# Trend data
obs_data = gistemp_trends_all["TAS_trends"].sel(RegionIndex=region_index, startyear=1)
obs_data = obs_data.rename({'duration':'year'})
obs_data = obs_data.assign_coords({'year':obs_data.year + 1900 - 1})

model_data = cesm1_pic_regional_trends_all.sel(RegionIndex=region_index)
model_data = model_data.rename({'duration':'year'})
model_data = model_data.assign_coords({'year':model_data.year + 1900 - 1})
region_name = obs_data.RegionName

ax = axs[1]

ax.fill_between(
    model_data.year,
    model_data.sel(percentile=0.025),
    model_data.sel(percentile=0.975),
    color='black',
    alpha=0.5,
    label='CESM1 Pre-Industrial 2$\sigma$',
    zorder=3,
)

ax.plot(
    obs_data.year,
    obs_data.median(dim="realization"),
    color='blue',
    label='GISTEMP Median',
    zorder=5,
)


for i in obs_data.realization:
    if i == 1:
        ax.plot(
            obs_data.year,
            obs_data.sel(realization=i),
            color='red',
            linewidth=0.5,
            alpha=0.2,
            label="GISTEMP Ensemble",
        )
    ax.plot(
        obs_data.year,
        obs_data.sel(realization=i),
        color='red',
        linewidth=0.5,
        alpha=0.2,
    )

ax.set_xlim(1900,2020)
ax.set_ylim(-0.05, 0.05)

ax.tick_params(axis='both', labelsize=18)

ax.set_ylabel('Surface Temperature Trend (K yr $^{-1}$)',fontsize=18)
ax.set_xlabel('Trend Endyear',fontsize=18)

ax.legend(fontsize=16)

to_png(file=fig,filename='thesis_methods2c',
       dpi=300,
       ext='pdf',
       bbox_inches='tight',
)
# %%

# Individual regions cone plots


fig, ax = plt.subplots(1, 1, figsize=(10,8))

# 17: 'West&Central-Europe'
# 26: 'E. Southern-Africa
region_index = 17
# 24 or 26

obs_data = gistemp_trends_all["TAS_trends"].sel(RegionIndex=region_index, startyear=1)
obs_data = obs_data.rename({'duration':'year'})
obs_data = obs_data.assign_coords({'year':obs_data.year + 1900 - 1})

model_data = cesm1_pic_regional_trends_all.sel(RegionIndex=region_index)
model_data = model_data.rename({'duration':'year'})
model_data = model_data.assign_coords({'year':model_data.year + 1900 - 1})
region_name = obs_data.RegionName

ax.fill_between(
    model_data.year,
    model_data.sel(percentile=0.025),
    model_data.sel(percentile=0.975),
    color='black',
    alpha=0.5,
    label='CESM1 Pre-Industrial 2$\sigma$',
    zorder=3,
)

ax.plot(
    obs_data.year,
    obs_data.median(dim="realization"),
    color='blue',
    label='GISTEMP Median',
    zorder=5,
)

for i in obs_data.realization:
    if i == 1:
        ax.plot(
            obs_data.year,
            obs_data.sel(realization=i),
            color='red',
            linewidth=0.5,
            alpha=0.2,
            label="GISTEMP Ensemble",
        )
    ax.plot(
        obs_data.year,
        obs_data.sel(realization=i),
        color='red',
        linewidth=0.5,
        alpha=0.2,
    )

ax.set_xlim(1900,2020)
ax.set_ylim(-0.05, 0.05)

ax.tick_params(axis='both', labelsize=18)

ax.set_ylabel('Surface Temperature Trend (K yr $^{-1}$)',fontsize=18)
ax.set_xlabel('Trend Endyear',fontsize=18)

ax.legend(fontsize=16)

to_png(file=fig,filename=f'thesis_methods_coneplot_{region_name.values}',
       dpi=300,
       ext='pdf',
       bbox_inches='tight',
)

# %%

## Old stuff

# %%

# Make figures

# Tibetan-Plateau region index
region_index = 34

obs_data = gistemp_5x5_ds[gistemp_tas_var].sel(RegionIndex=region_index)
obs_data = obs_data.rename({'recordlength':'year'})
obs_data = obs_data.assign_coords({'year':obs_data.year + 1900 - 1})

model_data = cesm1_pic_regional_trends_all.sel(RegionIndex=region_index)
model_data = model_data.rename({'duration':'year'})
model_data = model_data.assign_coords({'year':model_data.year + 1900 - 1})


region_name = obs_data.RegionName


fig, axs = plt.subplots(1,2,figsize=(20,8))
axs = axs.flat

plt.subplots_adjust(wspace=0.25)

# Plot the "deterministic" ensemble mean
axs[0].plot(
    obs_data.year,
    obs_data.median(dim='realization'),
    color='blue',
    alpha=1,
    label='GISTEMP',
)

for i in obs_data.realization:
    axs[1].plot(
        obs_data.year,
        obs_data.sel(realization=i),
        color='red',
        linewidth=0.5,
        alpha=0.1,
    )

# Adjust axes and labels
# even axes:
for _ax in axs:
    _ax.set_xlim(1900,2020)
    _ax.set_ylim(-1.6, 2)

    _ax.tick_params(axis='both', labelsize=18)

    _ax.set_ylabel('Surface Temperature Anomaly (K)',fontsize=18)
    _ax.set_xlabel('Year',fontsize=18)


to_png(file=fig,filename='thesis_methods1',
       dpi=300,
       ext='pdf',
       bbox_inches='tight',
)

# %%

# Tibetan-Plateau region index
region_index = 34

obs_data = gistemp_5x5_ds[gistemp_tas_var].sel(RegionIndex=region_index)
obs_data = obs_data.rename({'recordlength':'year'})
obs_data = obs_data.assign_coords({'year':obs_data.year + 1900 - 1})

model_data = cesm1_pic_regional_trends_all.sel(RegionIndex=region_index)
model_data = model_data.rename({'duration':'year'})
model_data = model_data.assign_coords({'year':model_data.year + 1900 - 1})

region_name = obs_data.RegionName


fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Plot the "deterministic" ensemble mean
ax.plot(
    obs_data.year,
    obs_data.median(dim='realization'),
    color='blue',
    alpha=1,
    label='GISTEMP',
    zorder=1000,
)

for i in obs_data.realization:
    ax.plot(
        obs_data.year,
        obs_data.sel(realization=i),
        color='red',
        linewidth=0.5,
        alpha=0.1,
    )

# Adjust axes and labels
# even axes:
ax.set_xlim(1900,2020)
ax.set_ylim(-1.6, 2)

ax.tick_params(axis='both', labelsize=18)

ax.set_ylabel('Surface Temperature Anomaly (K)',fontsize=18)
ax.set_xlabel('Year',fontsize=18)

to_png(file=fig,filename='thesis_methods1b',
       dpi=300,
       ext='pdf',
       bbox_inches='tight',
)

# %%
# Make second trends and emergence figure
fig, axs = plt.subplots(1,2,figsize=(20,8))
axs = axs.flat

obs_data = gistemp_trends_all["TAS_trends"].sel(RegionIndex=region_index, startyear=1)
obs_data = obs_data.rename({'duration':'year'})
obs_data = obs_data.assign_coords({'year':obs_data.year + 1900 - 1})

model_data = cesm1_pic_regional_trends_all.sel(RegionIndex=region_index)
model_data = model_data.rename({'duration':'year'})
model_data = model_data.assign_coords({'year':model_data.year + 1900 - 1})
region_name = obs_data.RegionName

axs[0].fill_between(
    model_data.year,
    model_data.sel(percentile=0.025),
    model_data.sel(percentile=0.975),
    color='black',
    alpha=0.5,
    label='CESM1 Pre-Industrial 2$\sigma$',
    zorder=3,
)

axs[0].plot(
    obs_data.year,
    obs_data.median(dim="realization"),
    color='blue',
    label='GISTEMP Median',
    zorder=5,
    # label="GISTEMP deterministic",
)

axs[1].fill_between(
    model_data.year,
    model_data.sel(percentile=0.025),
    model_data.sel(percentile=0.975),
    color='black',
    alpha=0.5,
    label='CESM1 Pre-Industrial 2$\sigma$',
    zorder=3,
)

for i in obs_data.realization:
    if i == 1:
        axs[1].plot(
            obs_data.year,
            obs_data.sel(realization=i),
            color='red',
            linewidth=0.5,
            alpha=0.2,
            label="GISTEMP Ensemble",
        )
    axs[1].plot(
        obs_data.year,
        obs_data.sel(realization=i),
        color='red',
        linewidth=0.5,
        alpha=0.2,
    )

# even axes:
for _ax in axs:
    _ax.set_xlim(1900,2020)
    _ax.set_ylim(-0.05, 0.05)
    # _ax.set_ylim(-0.2, 0.2)

    _ax.tick_params(axis='both', labelsize=18)

    _ax.set_ylabel('Surface Temperature Trend (K yr $^{-1}$)',fontsize=18)
    _ax.set_xlabel('Trend Endyear',fontsize=18)

axs[0].legend(fontsize=16)
axs[1].legend(fontsize=16)

to_png(file=fig,filename='thesis_methods2',
       dpi=300,
       ext='pdf',
       bbox_inches='tight',
)

# %%

fig, ax = plt.subplots(1, 1, figsize=(10,8))

obs_data = gistemp_trends_all["TAS_trends"].sel(RegionIndex=region_index, startyear=1)
obs_data = obs_data.rename({'duration':'year'})
obs_data = obs_data.assign_coords({'year':obs_data.year + 1900 - 1})

model_data = cesm1_pic_regional_trends_all.sel(RegionIndex=region_index)
model_data = model_data.rename({'duration':'year'})
model_data = model_data.assign_coords({'year':model_data.year + 1900 - 1})
region_name = obs_data.RegionName

ax.fill_between(
    model_data.year,
    model_data.sel(percentile=0.025),
    model_data.sel(percentile=0.975),
    color='black',
    alpha=0.5,
    label='CESM1 Pre-Industrial 2$\sigma$',
    zorder=3,
)

ax.plot(
    obs_data.year,
    obs_data.median(dim="realization"),
    color='blue',
    label='GISTEMP Median',
    zorder=5,
    # label="GISTEMP deterministic",
)


for i in obs_data.realization:
    if i == 1:
        ax.plot(
            obs_data.year,
            obs_data.sel(realization=i),
            color='red',
            linewidth=0.5,
            alpha=0.2,
            label="GISTEMP Ensemble",
        )
    ax.plot(
        obs_data.year,
        obs_data.sel(realization=i),
        color='red',
        linewidth=0.5,
        alpha=0.2,
    )

ax.set_xlim(1900,2020)
ax.set_ylim(-0.05, 0.05)

ax.tick_params(axis='both', labelsize=18)

ax.set_ylabel('Surface Temperature Trend (K yr $^{-1}$)',fontsize=18)
ax.set_xlabel('Trend Endyear',fontsize=18)

ax.legend(fontsize=16)

to_png(file=fig,filename='thesis_methods2b',
       dpi=300,
       ext='pdf',
       bbox_inches='tight',
)