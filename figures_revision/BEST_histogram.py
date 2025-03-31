"""
To demonstrate the ToE for BEST, I will compare it to the median ToE of the other observational datasets.
The plots should resemble Figure 1 of the main paper.
"""

# %%
import os

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import regionmask
import seaborn as sns
import copy

# %%


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


def convert_ipcc_to_latlon(
    data: xr.DataArray,
    mask: regionmask.Regions = regionmask.defined_regions.ar6.all,
    metric_list: list = ['mean', 'strict', 'delay', 'delay_mask'],
):

    metric_data_list = []
    for metric in metric_list:
        if metric not in data.metric:
            raise ValueError(f"Metric '{metric}' not found in the data.")
        metric_data_list.append(data.sel(metric=metric))

    unc_mask = mask.mask(np.arange(-179.5, 180, 0.5),np.arange(-89.5, 90, 0.5),)

    metric_mask_list = [unc_mask.copy() for i in metric_data_list]  # Create a list to hold the masks for each metric

    del unc_mask

    offset = -1000 # Need an offset or it may confuse regions with the delays (delays are on [0,40] similar to the region indices)
    lat_lon_data_list = []
    for i in mask.numbers:
        
        for j, (metric_data, metric_mask) in enumerate(zip(metric_data_list, metric_mask_list)):
            # Update the mask with the metric values.
            metric_mask = xr.where(metric_mask==i, metric_data.sel(RegionIndex=i) + offset, metric_mask)
            metric_mask_list[j] = metric_mask  # Update the mask in the list


    regional_masked_da = xr.concat(metric_mask_list, dim='metric') - offset
    return regional_masked_da

# %%

## Load computed ToE data
toe_savedir = "/glade/u/home/jonahshaw/w/trend_uncertainty/nathan/ToE_output"

spatial_masked_da = xr.open_dataarray(f"{toe_savedir}/ToE_delay.5x5.nc")
regional_masked_da = xr.open_dataarray(f"{toe_savedir}/ToE_delay.IPCCRegions.nc")

all_spatial_da = xr.open_dataarray(f"{toe_savedir}/all_ToE.5x5.nc")
all_regional_da = xr.open_dataarray(f"{toe_savedir}/all_ToE.IPCCRegions.nc")

# %%

# Process the all combinations data.
# Spatial data
# all_spatial_obs_da = all_spatial_da.sel(obs="BEST_any")
all_spatial_obs_da = all_spatial_da.drop_sel(obs=["DCENT_unfilled", "BEST", "BEST_any"])

strict_spatial_toe = all_spatial_obs_da.where(~np.isnan(all_spatial_obs_da),np.inf).quantile(q=0.95, dim=['realization', "model", "obs"], method='nearest')
strict_spatial_toe = strict_spatial_toe.where(~np.isinf(strict_spatial_toe),np.nan).drop_vars("quantile")
strict_spatial_toe = strict_spatial_toe.squeeze().assign_coords({'metric':'strict'},).expand_dims(['metric'])

mean_spatial_toe = all_spatial_obs_da.where(~np.isnan(all_spatial_obs_da),np.inf).quantile(q=0.5, dim=['realization', "model", "obs"], method='nearest')
mean_spatial_toe = mean_spatial_toe.where(~np.isinf(mean_spatial_toe),np.nan).drop_vars("quantile")
mean_spatial_toe = mean_spatial_toe.assign_coords({'metric':'mean'},).expand_dims(['metric'])

# Compute the delay and conditional delay mask:
# delay_toe = strict_spatial_toe - mean_spatial_toe # Compute the delay
# delay_undef_region = np.bitwise_and(np.isnan(strict_spatial_toe), ~np.isnan(mean_spatial_toe)) # Mask for where tas is emerged in the mean but not the "strict" sense.
# delay_toe = xr.where(delay_undef_region, 2020 - mean_spatial_toe, delay_toe) # Apply the conditional mask

# all_spatial_toes = xr.concat(
#     [mean_spatial_toe.assign_coords({'metric':'mean'},).expand_dims(['metric'],axis=[-1]),
#     strict_spatial_toe.assign_coords({'metric':'strict'},).expand_dims(['metric'],axis=[-1]),
#     delay_toe.assign_coords({'metric':'delay'},).expand_dims(['metric'],axis=[-1]),
#     delay_undef_region.assign_coords({'metric':'delay_mask'},).expand_dims(['metric'],axis=[-1]),
#     ],
#     dim='metric',
# )

# all_spatial_toes = all_spatial_toes.assign_coords({'obs':'all', "model":"all"}).expand_dims(['obs',"model"])

# %%

# Compute ToE in the most conservative sense using the 4 models and the HadCRUT and GISTEMP obs.
all_regional_obs_da = all_regional_da.drop_sel(obs=["DCENT_unfilled", "BEST", "BEST_any"])

# Set nans to inf before computing quantiles, then set back to nan.
strict_regional_toe = all_regional_obs_da.where(~np.isnan(all_regional_obs_da), np.inf).quantile(q=0.95, dim=['realization', "model", "obs"], method='nearest')
strict_regional_toe = strict_regional_toe.where(~np.isinf(strict_regional_toe), np.nan).drop_vars("quantile")
strict_regional_toe = strict_regional_toe.squeeze().assign_coords({'metric':'strict'},).expand_dims(['metric'])

mean_regional_toe = all_regional_obs_da.where(~np.isnan(all_regional_obs_da), np.inf).quantile(q=0.5, dim=['realization', "model", "obs"], method='nearest')
mean_regional_toe = mean_regional_toe.where(~np.isinf(mean_regional_toe), np.nan).drop_vars("quantile")
mean_regional_toe = mean_regional_toe.assign_coords({'metric':'mean'},).expand_dims(['metric'])
# Compute the delay and conditional delay mask:
# delay_toe = strict_regional_toe - mean_regional_toe # Compute the delay
# delay_undef_region = np.bitwise_and(np.isnan(strict_regional_toe), ~np.isnan(mean_regional_toe)) # Mask for where tas is emerged in the mean but not the "strict" sense.
# delay_toe = xr.where(delay_undef_region, 2020 - mean_regional_toe, delay_toe) # Apply the conditional mask

# all_regional_toes = xr.concat(
#     [mean_regional_toe.assign_coords({'metric':'mean'},).expand_dims(['metric'],axis=[-1]),
#     strict_regional_toe.assign_coords({'metric':'strict'},).expand_dims(['metric'],axis=[-1]),
#     delay_toe.assign_coords({'metric':'delay'},).expand_dims(['metric'],axis=[-1]),
#     delay_undef_region.assign_coords({'metric':'delay_mask'},).expand_dims(['metric'],axis=[-1]),
#     ],
#     dim='metric',
# )

# all_regional_toes = all_regional_toes.assign_coords({'obs':'all', "model":"all"}).expand_dims(['obs',"model"])

# %%

# Make a better land mask for the spatial 5x5 deg. regions
land = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask_3D_frac_approx(
    spatial_masked_da
).squeeze()
spatial_landmask = land>0.5
# Need to convert the IPCC Region Indexing to lat-lon coordinates in order to weight by area.
# Apply the land mask here.
mask = regionmask.defined_regions.ar6.land
regional_latlon_da = convert_ipcc_to_latlon(
    regional_masked_da.drop_sel(metric=['delay', 'delay_mask']).drop_sel(obs="DCENT_unfilled").compute(),
    metric_list=["mean", "strict"],
)

# Do for all of the data (e.g. the combine 1600 realizations)
regional_latlon_all_da = convert_ipcc_to_latlon(
    strict_regional_toe.compute(),
    metric_list=["strict"],
    # metric_list=["mean"],
)

landmask = ~np.isnan(mask.mask(regional_latlon_da["lon"], regional_latlon_da["lat"],))

# %%
# Data is BEST_any:
# - Start years can be pre-1900. 
# - All four models are used and the most conservative estimate is used.
regional_data_latloned1 = regional_latlon_da.sel(obs="BEST_any", metric="mean").max(dim="model", skipna=False).expand_dims(['metric'])
spatial_data1  = all_spatial_da.squeeze().sel(obs="BEST_any", realization=0).max(dim="model", skipna=False).assign_coords({'metric':'mean'},).expand_dims(['metric'])

regional_data_latloned2 = regional_latlon_all_da
spatial_data2 = strict_spatial_toe

# Set up masks!
land  = regionmask.defined_regions.natural_earth_v5_0_0.land_110
ocean = regionmask.defined_regions.natural_earth_v5_0_0.ocean_basins_50

# Pick your mask
mask = land #None, land, ocean

fig,axs = plt.subplots(1, 2, figsize=(12,6))
fig.subplots_adjust(wspace=0.35)
axs = axs.flat

# startyear = 1900
startyear = 1880

# Start plot 1
ax = axs[0]
mean_data = regional_data_latloned1
strict_data = regional_data_latloned2
zstack_vars = ('lat','lon')#,'model','obs')

mean_data_subset0    = copy.copy(mean_data.sel(metric='mean'))
strict_data_subset0  = copy.copy(strict_data.sel(metric='strict'))

nan_mask_value  = 2027.5
nan_mask_value2 = 2036

# Set nans to the different masking values out of the normal range (1900, 2020)
mean_data_subset    = xr.where(np.isnan(mean_data_subset0),nan_mask_value,mean_data_subset0)
mean_data_subset2   = xr.where(np.isnan(mean_data_subset0),nan_mask_value2,mean_data_subset0)
strict_data_subset  = xr.where(np.isnan(strict_data_subset0),nan_mask_value,strict_data_subset0)
strict_data_subset2 = xr.where(np.isnan(strict_data_subset0),nan_mask_value2,strict_data_subset0)

ax2 = ax.twinx()

bins = np.arange(startyear + 0.01, nan_mask_value2+5, 5)
xticks = np.arange(startyear, nan_mask_value2+5, 5)
xticks = list(np.arange(startyear, 2021, 10))
xticks.append(nan_mask_value)
xticklabels = list(np.arange(startyear, 2021, 10))
xticklabels.append('Not \n Emerged')
yticks = np.arange(0, 0.21, 0.05)

# This should work to do the weighting but it doesn't!
mean_data_weights,_     = xr.broadcast(np.cos(np.deg2rad(mean_data_subset.lat)),mean_data_subset)
mean_data_weights_stack = mean_data_weights.stack(z=zstack_vars).values
mean_data_subset_stack  = mean_data_subset.stack(z=zstack_vars).values
mean_data_subset_stack2 = mean_data_subset2.stack(z=zstack_vars).values

# Set up mask with the data dimensions and do masking on the stacked data (cannot do boolean indexing over 4 dimensions)
if (not mask is None):
    mask_broad = np.isfinite(mask.mask(mean_data.lon, mean_data.lat)).broadcast_like(mean_data.isel(metric=0))
    mask_broad_stack        = mask_broad.stack(z=zstack_vars)

    mean_data_weights_stack = mean_data_weights_stack[mask_broad_stack]
    mean_data_subset_stack = mean_data_subset_stack[mask_broad_stack]
    mean_data_subset_stack2 = mean_data_subset_stack2[mask_broad_stack]

_n,_bins,_pathes = ax.hist(
    x=mean_data_subset_stack,
    bins=bins,
    color='red',alpha=0.5,label='BEST',
    density=False,
    weights=mean_data_weights_stack/mean_data_weights_stack.sum(),
)

_n,_bins,_pathes = ax2.hist(
    x=mean_data_subset_stack2,
    bins=bins,
    color='darkred',alpha=0.8,label='Median ToE Cumulative',
    density=False,
    weights=mean_data_weights_stack/mean_data_weights_stack.sum(),
    histtype="step",        
    cumulative=True
)

print('Regional Median fraction emerge: %.3f' % _n[-2])

strict_data_weights,_     = xr.broadcast(np.cos(np.deg2rad(strict_data_subset.lat)),strict_data_subset)
strict_data_weights_stack = strict_data_weights.stack(z=zstack_vars).values
strict_data_subset_stack  = strict_data_subset.stack(z=zstack_vars).values
strict_data_subset_stack2 = strict_data_subset2.stack(z=zstack_vars).values

# Set up mask with the data dimensions and do masking on the stacked data (cannot do boolean indexing over 4 dimensions)
if (not mask is None):
    mask_broad = np.isfinite(mask.mask(strict_data.lon, strict_data.lat)).broadcast_like(strict_data.isel(metric=0))
    mask_broad_stack        = mask_broad.stack(z=zstack_vars)

    strict_data_weights_stack = strict_data_weights_stack[mask_broad_stack]
    strict_data_subset_stack = strict_data_subset_stack[mask_broad_stack]
    strict_data_subset_stack2 = strict_data_subset_stack2[mask_broad_stack]

_n,_bins,_pathes = ax.hist(
    x=strict_data_subset_stack,
    bins=bins,
    color='blue',alpha=0.5,label='Strict',
    density=False,
    weights=strict_data_weights_stack/strict_data_weights_stack.sum(),
)

_n,_bins,_pathes = ax2.hist(
    x=strict_data_subset_stack2,
    bins=bins,
    color='darkblue',alpha=0.8,label='Strict ToE Cumulative',
    density=False,
    # weights=mean_data_weights_stack/mean_data_weights_stack.sum(),
    weights=strict_data_weights_stack/strict_data_weights_stack.sum(),
    histtype="step",
    cumulative=True
)

print('Regional Strict fraction emerge: %.3f' % _n[-2])

# Ax1 set-up
ax.set_xticks(xticks)
ax.set_xlim(startyear, 2032.5)
# ax.set_xlim(1967.5,2032.5)
ax.set_xticklabels(xticklabels,rotation=45)
ax.tick_params(axis='both',labelsize=12)
ax.set_xlabel('Time of Emergence',fontsize=16)

ax.set_yticks(yticks)
ax.set_ylabel('Land Area Fraction',fontsize=14)

ax.set_title('IPCC Regions',fontsize=18)
ax.legend(fontsize=12,loc='upper left')

# Ax2 set-up
ax2.tick_params(axis='both',labelsize=12)

ax2.set_ylim(0,1.0)
ax2.set_ylabel('Cumulative Land Area Fraction',fontsize=14)
_ = ax2.set_yticks(np.arange(0,1.1,0.2))

# Subplot 2
ax = axs[1]
mean_data = spatial_data1
strict_data = spatial_data2

mean_data_subset0 = copy.copy(mean_data.sel(metric='mean'))
strict_data_subset0  = copy.copy(strict_data.sel(metric='strict'))

mean_data_subset    = xr.where(np.isnan(mean_data_subset0),nan_mask_value,mean_data_subset0)
mean_data_subset2   = xr.where(np.isnan(mean_data_subset0),nan_mask_value2,mean_data_subset0)
strict_data_subset  = xr.where(np.isnan(strict_data_subset0),nan_mask_value,strict_data_subset0)
strict_data_subset2 = xr.where(np.isnan(strict_data_subset0),nan_mask_value2,strict_data_subset0)

ax2 = ax.twinx()

bins = np.arange(startyear + 0.01, 2041, 5)
xticks = list(np.arange(startyear, 2021, 10))
xticks.append(nan_mask_value)
xticklabels = list(np.arange(startyear, 2021, 10))
xticklabels.append('Not \n Emerged')
yticks = np.arange(0, 0.21, 0.05)

# This should work to do the weighting but it doesn't!
mean_data_weights,_ = xr.broadcast(np.cos(np.deg2rad(mean_data_subset.lat)),mean_data_subset)
mean_data_weights_stack = mean_data_weights.stack(z=zstack_vars)
mean_data_subset_stack  = mean_data_subset.stack(z=zstack_vars)
mean_data_subset_stack2 = mean_data_subset2.stack(z=zstack_vars)

# Set up mask with the data dimensions and do masking on the stacked data (cannot do boolean indexing over 4 dimensions)
if (not mask is None):
    mask_broad = np.isfinite(mask.mask(mean_data.lon, mean_data.lat)).broadcast_like(mean_data.isel(metric=0))
    mask_broad_stack        = mask_broad.stack(z=zstack_vars)

    mean_data_weights_stack = mean_data_weights_stack[mask_broad_stack]
    mean_data_subset_stack = mean_data_subset_stack[mask_broad_stack]
    mean_data_subset_stack2 = mean_data_subset_stack2[mask_broad_stack]

_n,_bins,_pathes = ax.hist(
    x=np.array(mean_data_subset_stack),
    bins=bins,
    color='red',alpha=0.5,label='BEST',
    density=False,
    weights=np.array(mean_data_weights_stack/mean_data_weights_stack.sum()),
)

_n,_bins,_pathes = ax2.hist(
    x=np.array(mean_data_subset_stack2),
    bins=bins,
    color='darkred',alpha=0.8,label='Median ToE Cumulative',
    density=True,
    weights=np.array(mean_data_weights_stack/mean_data_weights_stack.sum()),
    histtype="step",        
    cumulative=True
)

print('5x5 deg. Median fraction emerge: %.3f' % _n[-2])

strict_data_weights,_ = xr.broadcast(np.cos(np.deg2rad(strict_data_subset.lat)),strict_data_subset)
strict_data_weights_stack = strict_data_weights.stack(z=zstack_vars)
strict_data_subset_stack  = strict_data_subset.stack(z=zstack_vars)
strict_data_subset_stack2 = strict_data_subset2.stack(z=zstack_vars)

# Set up mask with the data dimensions and do masking on the stacked data (cannot do boolean indexing over 4 dimensions)
if (not mask is None):
    mask_broad = np.isfinite(mask.mask(strict_data.lon, strict_data.lat)).broadcast_like(strict_data.isel(metric=0))
    mask_broad_stack        = mask_broad.stack(z=zstack_vars)

    strict_data_weights_stack = strict_data_weights_stack[mask_broad_stack]
    strict_data_subset_stack = strict_data_subset_stack[mask_broad_stack]
    strict_data_subset_stack2 = strict_data_subset_stack2[mask_broad_stack]

_n,_bins,_pathes = ax.hist(
    x=np.array(strict_data_subset_stack),
    bins=bins,
    color='blue',alpha=0.5,label='Strict',
    density=False,
    weights=np.array(strict_data_weights_stack/strict_data_weights_stack.sum()),
)

_n,_bins,_pathes = ax2.hist(
    x=np.array(strict_data_subset_stack2),
    bins=bins,
    color='darkblue',alpha=0.8,label='Strict ToE Cumulative',
    density=False,
    weights=np.array(strict_data_weights_stack/strict_data_weights_stack.sum()),
    histtype="step",        
    cumulative=True
)

print('5x5 deg. Strict fraction emerge: %.3f' % _n[-2])

# Ax1 set-up
ax.set_xticks(xticks)
ax.set_xlim(startyear, 2032.5)
ax.set_xticklabels(xticklabels,rotation=45)
ax.tick_params(axis='both',labelsize=12)
ax.set_xlabel('Time of Emergence',fontsize=16)

ax.set_yticks(yticks)
ax.set_ylabel('Land Area Fraction',fontsize=14)

ax.set_title('$5^{\circ}$x$5^{\circ}$ Regions',fontsize=18)
ax.legend(fontsize=12,loc='upper left')

# Ax2 set-up
ax2.tick_params(axis='both',labelsize=12)

ax2.set_ylim(0,1.0)
ax2.set_ylabel('Cumulative Land Area Fraction',fontsize=14)
_ = ax2.set_yticks(np.arange(0, 1.1, 0.2))

# Add subplot labels
axs[0].text(-0.1, 1.08, 'a.', transform=axs[0].transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
axs[1].text(-0.1, 1.08, 'b.', transform=axs[1].transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

# %%