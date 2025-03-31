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
import cartopy.crs as ccrs

from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

# %%

def sp_map(*nrs, projection = ccrs.PlateCarree(), **kwargs):
    return plt.subplots(*nrs, subplot_kw={'projection':projection}, **kwargs)


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
):

    mean_data   = data.sel(metric='mean')
    strict_data = data.sel(metric='strict')
    delay_data  = data.sel(metric='delay')
    delay_mask  = data.sel(metric='delay_mask')

    unc_mask = mask.mask(np.arange(-179.5, 180, 0.5),np.arange(-89.5, 90, 0.5),)

    # CESM1-GISTEMP 5x5
    _mean_mask   = unc_mask.copy()
    _strict_mask = unc_mask.copy()
    _delay_data_mask  = unc_mask.copy()
    _delay_mask_mask  = unc_mask.copy()

    del unc_mask

    offset = -1000 # Need an offset or it may confuse regions with the delays (delays are on [0,40] similar to the region indices)
    for i in mask.numbers:
        
        _mean_mask   = xr.where(_mean_mask==i,mean_data.sel(RegionIndex=i)+offset,_mean_mask)
        _strict_mask = xr.where(_strict_mask==i,strict_data.sel(RegionIndex=i)+offset,_strict_mask)
        _delay_data_mask  = xr.where(_delay_data_mask==i,delay_data.sel(RegionIndex=i)+offset,_delay_data_mask)
        _delay_mask_mask  = xr.where(_delay_mask_mask==i,delay_mask.sel(RegionIndex=i)+offset,_delay_mask_mask)

    regional_masked_da = xr.concat([_mean_mask,_strict_mask,_delay_data_mask,_delay_mask_mask],dim='metric')-offset
    
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
all_spatial_obs_da = all_spatial_da.drop_sel(obs="DCENT_unfilled")

strict_spatial_toe = all_spatial_obs_da.where(~np.isnan(all_spatial_obs_da),np.inf).quantile(q=0.95, dim=['realization', "model", "obs"], method='nearest')
strict_spatial_toe = strict_spatial_toe.where(~np.isinf(strict_spatial_toe),np.nan).drop_vars("quantile")

mean_spatial_toe = all_spatial_obs_da.where(~np.isnan(all_spatial_obs_da),np.inf).quantile(q=0.5, dim=['realization', "model", "obs"], method='nearest')
mean_spatial_toe = mean_spatial_toe.where(~np.isinf(mean_spatial_toe),np.nan).drop_vars("quantile")

# Compute the delay and conditional delay mask:
delay_toe = strict_spatial_toe - mean_spatial_toe # Compute the delay
delay_undef_region = np.bitwise_and(np.isnan(strict_spatial_toe), ~np.isnan(mean_spatial_toe)) # Mask for where tas is emerged in the mean but not the "strict" sense.
delay_toe = xr.where(delay_undef_region, 2020 - mean_spatial_toe, delay_toe) # Apply the conditional mask

all_spatial_toes = xr.concat(
    [mean_spatial_toe.assign_coords({'metric':'mean'},).expand_dims(['metric'],axis=[-1]),
    strict_spatial_toe.assign_coords({'metric':'strict'},).expand_dims(['metric'],axis=[-1]),
    delay_toe.assign_coords({'metric':'delay'},).expand_dims(['metric'],axis=[-1]),
    delay_undef_region.assign_coords({'metric':'delay_mask'},).expand_dims(['metric'],axis=[-1]),
    ],
    dim='metric',
)

all_spatial_toes = all_spatial_toes.assign_coords({'obs':'all', "model":"all"}).expand_dims(['obs',"model"])

# Regional Data
all_regional_obs_da = all_regional_da.drop_sel(obs="DCENT_unfilled")

strict_regional_toe = all_regional_obs_da.where(~np.isnan(all_regional_obs_da), np.inf).quantile(q=0.95, dim=['realization', "model", "obs"], method='nearest')
strict_regional_toe = strict_regional_toe.where(~np.isinf(strict_regional_toe), np.nan).drop_vars("quantile")

mean_regional_toe = all_regional_obs_da.where(~np.isnan(all_regional_obs_da), np.inf).quantile(q=0.5, dim=['realization', "model", "obs"], method='nearest')
mean_regional_toe = mean_regional_toe.where(~np.isinf(mean_regional_toe), np.nan).drop_vars("quantile")

# Compute the delay and conditional delay mask:
delay_toe = strict_regional_toe - mean_regional_toe # Compute the delay
delay_undef_region = np.bitwise_and(np.isnan(strict_regional_toe), ~np.isnan(mean_regional_toe)) # Mask for where tas is emerged in the mean but not the "strict" sense.
delay_toe = xr.where(delay_undef_region, 2020 - mean_regional_toe, delay_toe) # Apply the conditional mask

all_regional_toes = xr.concat(
    [mean_regional_toe.assign_coords({'metric':'mean'},).expand_dims(['metric'],axis=[-1]),
    strict_regional_toe.assign_coords({'metric':'strict'},).expand_dims(['metric'],axis=[-1]),
    delay_toe.assign_coords({'metric':'delay'},).expand_dims(['metric'],axis=[-1]),
    delay_undef_region.assign_coords({'metric':'delay_mask'},).expand_dims(['metric'],axis=[-1]),
    ],
    dim='metric',
)

all_regional_toes = all_regional_toes.assign_coords({'obs':'all', "model":"all"}).expand_dims(['obs',"model"])


# %%

# Make a better land mask for the spatial 5x5 deg. regions
land = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask_3D_frac_approx(
    spatial_masked_da
).squeeze()
spatial_landmask = land>0.5
# Need to convert the IPCC Region Indexing to lat-lon coordinates in order to plot spatially.
# Create a landmaks for the IPCC regions.
mask = regionmask.defined_regions.ar6.land
regional_latlon_da = convert_ipcc_to_latlon(
    regional_masked_da,
    mask=mask,
)

# Do for all of the data
regional_latlon_all_da = convert_ipcc_to_latlon(
    all_regional_toes,
    mask=mask,
)

landmask = ~np.isnan(mask.mask(regional_latlon_da["lon"], regional_latlon_da["lat"],))

# %%

# Function for the regional figures.
def plot_regional_toe_and_delay(
    data,
    obs_list,
    model,
    toe_levels=np.linspace(1960,2020,13),
    labels=None,
    surfacemask=None,
    surfacehatch="o",
):

    data_subset     = data.sel(obs=obs_list,model=model)
    data_mean       = data_subset.sel(metric='mean')

    fig,axs = sp_map(1,3,projection=ccrs.Robinson(),figsize=(14,4))
    axs = axs.flat

    cax1 = plt.axes([0.24, 0.22, 0.56, 0.03])

    ToE_cmap = copy.copy(mpl.cm.viridis)
    ToE_cmap.set_over('white')
    
    delay_cmap = sns.light_palette('red',n_colors=10,reverse=False,as_cmap=True)
    delay_cmap.set_over('magenta')
    
    masks = [data_mean.sel(obs=i) for i in data_mean.obs.values]
    
    if labels is None:
        labels = data_mean.obs.values

    _levels = toe_levels
    _cmap = ToE_cmap
    _extend = 'max'

    plt.subplots_adjust(wspace=0.05)

    ims = [] 
    for _ax,_mask,_label in zip(axs,masks,labels):

        im = _ax.contourf(_mask.lon,_mask.lat,_mask, 60,
                          transform=ccrs.PlateCarree(),
                          levels=_levels,
                          cmap=_cmap,
                          extend=_extend,
                         )

        _ax.coastlines()
        _ax.set_global()
        _ax.set_title(_label,fontsize=16)

        ims.append(im)

        _ = _ax.contourf(
            surfacemask.lon,
            surfacemask.lat,
            ~surfacemask,
            levels=[0,0.5,2],
            colors='none',
            transform=ccrs.PlateCarree(),
            hatches=[None, surfacehatch,],
            extend='max',
        )

    cbar1 = fig.colorbar(ims[0],orientation='horizontal',
                         ticks=toe_levels,
                         cax=cax1,
                        )
    cbar1.ax.tick_params(labelsize=12)
    
    fig.text(0.14, 0.68, "a.", fontsize=14)
    fig.text(0.40, 0.68, "b.", fontsize=14)
    fig.text(0.665, 0.68, "c.", fontsize=14)
    
    return fig


def plot_spatial_toe_and_delay(
    data,
    obs_list,
    model,
    toe_levels=np.linspace(1960,2020,13),
    labels=None,
    surfacemask=None,
    surfacehatch="o",
):

    data_subset     = data.sel(obs=obs_list,model=model)
    data_mean       = data_subset.sel(metric='mean')

    fig,axs = sp_map(1,3,projection=ccrs.Robinson(),figsize=(14,4))
    axs = axs.flat

    cax1 = plt.axes([0.24, 0.22, 0.56, 0.03])

    ToE_cmap = copy.copy(mpl.cm.viridis)
    ToE_cmap.set_over('white')
    
    delay_cmap = sns.light_palette('red',n_colors=10,reverse=False,as_cmap=True)
    delay_cmap.set_over('magenta')
    delay_cmap.colorbar_extend = True

    masks = [data_mean.sel(obs=i) for i in data_mean.obs.values]
    
    if labels is None:
        labels = data_mean.obs.values

    _levels = toe_levels
    _cmap = ToE_cmap

    plt.subplots_adjust(wspace=0.05)

    ims = [] 
    for _ax,_mask,_label in zip(axs,masks,labels):

        _norm = BoundaryNorm(_levels, ncolors=_cmap.N, clip=False)

        im = _ax.pcolormesh(
            _mask.lon,
            _mask.lat,
            _mask.squeeze(),
            transform=ccrs.PlateCarree(),
            cmap=_cmap,
            norm=_norm, 
        )

        _ax.coastlines()
        _ax.set_global()
        _ax.set_title(_label,fontsize=16)

        ims.append(im)

        if surfacemask is not None:
            _ = _ax.contourf(
                surfacemask.lon,
                surfacemask.lat,
                ~surfacemask,
                levels=[0,0.5,2],
                colors='none',
                transform=ccrs.PlateCarree(),
                hatches=[None, surfacehatch,],
                extend='max',
           )

    cbar1 = fig.colorbar(ims[0],orientation='horizontal',
                         ticks=toe_levels,
                         cax=cax1,
                         extend='max',
                        )
    cbar1.ax.tick_params(labelsize=12)

    fig.text(0.14, 0.68, "d.", fontsize=14)
    fig.text(0.40, 0.68, "e.", fontsize=14)
    fig.text(0.665, 0.68, "f.", fontsize=14)

    return fig


# %%

out = plot_regional_toe_and_delay(
    data=regional_latlon_da,
    obs_list=['GISTEMP_5x5', 'HadCRUT', 'BEST'],
    model='CESM1',
    toe_levels=np.linspace(1920,2020,11),
    labels=['GISTEMP','HadCRUT','BEST'],
    surfacemask=landmask,
    surfacehatch='..',
)

to_png(
    out, 
    filename='median_ToEs_BEST_regional',
    dpi=300,
    bbox_inches='tight',
    ext="pdf",
    # pad_inches=0.1,
)
# %%

out = plot_spatial_toe_and_delay(
    data=spatial_masked_da.where(spatial_landmask),  # Only plot where the land mask is true.
    obs_list=['GISTEMP_5x5', 'HadCRUT', 'BEST'],
    model='CESM1',
    toe_levels=np.linspace(1920,2020,11),
    labels=['GISTEMP','HadCRUT','BEST'],
    surfacemask=spatial_landmask,
    surfacehatch='..',
)

to_png(
    out, 
    filename='median_ToEs_BEST_spatial',
    dpi=300,
    bbox_inches='tight',
    ext="pdf",
    # pad_inches=0.1,
)
# %%
