# Define a function to build the normalisations depending on the strategy -- should be moved to a separate Python function file
def build_normalisation(file, norm_strat, norm_variables, time_window, train_end, mask, area_weight=True, mask_var_index=1):
    #Load the xarray data and time/variable slice it. 
    ds_stats = xr.open_dataset(file)
    times = pd.date_range(**time_window)
    ds_stats = ds_stats.sel(time=times, method = "nearest")
    area_t = ds_stats.area_t
    ds_stats = ds_stats[norm_variables]

    # Load the mask 
    mask_var = norm_variables[mask_var_index]
    warnings.warn(f"Assuming {mask_var} has NaNs at t=0 to indicate land mask.")
    mask_xr = xr.where(np.isnan(ds_stats[mask_var].isel(time=0)), 0, 1)
    if mask:
        masked_stats = ds_stats.where(mask_xr == 1)
    else: 
        warnings.warn("Normalising without taking into account land mask")
        masked_stats = ds_stats

    # Set area weighting:
    if area_weight:
        aweight = area_t.where(mask_xr == 1).fillna(0.).load()
    else:
        aweight = xr.ones_like(area_t).load()
        
    # Align spatial dimension names with the pipeline (Sort uses latitude/longitude).
    rename_map = {}
    if "yt_ocean" in masked_stats.dims:
        rename_map["yt_ocean"] = "latitude"
    if "xt_ocean" in masked_stats.dims:
        rename_map["xt_ocean"] = "longitude"
    if rename_map:
        masked_stats = masked_stats.rename(rename_map)
        aweight = aweight.rename(rename_map)

    # The below if statements define the mean and stds. 
    # To avoid data leakage, only the training data is used for mean/std calculation
    if norm_strat == "Global_in_time":
        # 1) Global stats (scalar per variable)
        mean_stats = ((masked_stats.sel(time=slice(times[0], train_end))*aweight).sum(dim=("latitude", "longitude"), skipna=True)/aweight.sum()).mean('time')
        std_stats = (masked_stats.sel(time=slice(times[0], train_end)).std('time')*aweight).sum(dim=("latitude", "longitude"), skipna=True)/aweight.sum()
        std_stats = std_stats.drop_vars(['time'])
    elif norm_strat == "Spatial_in_time":
        # 2) Spatial stats (per-gridcell over all times)
        mean_stats = masked_stats.sel(time=slice(times[0], train_end)).mean(dim="time", skipna=True)
        std_stats = masked_stats.sel(time=slice(times[0], train_end)).std(dim="time", skipna=True)
    elif norm_strat == "Spatial_climatology":
        # 3) Spatial climatology (month-wise per gridcell) then tile over all years.
        mean_stats = masked_stats.sel(time=slice(times[0], train_end)).groupby("time.month").mean(dim="time", skipna=True)
        std_stats = (masked_stats.sel(time=slice(times[0], train_end)).groupby("time.month")\
                     - mean_stats).groupby('time.month').std(dim="time", skipna=True)

        month_indexer = xr.DataArray(
            masked_stats.time.dt.month.values,
            coords={"time": masked_stats.time.values},
            dims="time",
        )
        mean_stats = mean_stats.sel(month=month_indexer).drop_vars("month", errors="ignore")
        std_stats = std_stats.sel(month=month_indexer).drop_vars("month", errors="ignore")
    elif norm_strat == "Spatial_climatology_global_variance":
        # 4) Spatial climatology for mean, but do standard deviation globally
        mean_stats = masked_stats.sel(time=slice(times[0], train_end)).groupby("time.month").mean(dim="time", skipna=True)
        std_stats = ((masked_stats.sel(time=slice(times[0], train_end)).groupby("time.month")\
                     - mean_stats).std("time")*aweight).sum(dim=("latitude", "longitude"), skipna=True)/aweight.sum()

        month_indexer = xr.DataArray(
            masked_stats.time.dt.month.values,
            coords={"time": masked_stats.time.values},
            dims="time",
        )
        mean_stats = mean_stats.sel(month=month_indexer).drop_vars("month", errors="ignore")
        std_stats = std_stats.drop_vars(['time'])
    else: 
        raise ValueError("No valid normlaisation strategy provided. Options are Global_in_time, Spatial_in_time, and Spatial_climatology")

    # Transform mean and std to Datasets so co-ordinates align with the raw data
    mean_stats_ds = xr.Dataset({var: mean_stats[var] for var in norm_variables})
    std_stats_ds  = xr.Dataset({var: std_stats[var]  for var in norm_variables})

    # Define pipeline normalisation operations for the strategy.
    pipelines_normed = petpipe.operations.xarray.normalisation.Evaluated(
                        normalisation_eval="(sample - mean) / deviation",
                        unnormalisation_eval="(sample * deviation) + mean",
                        mean = mean_stats_ds,
                        deviation = std_stats_ds)
    
    return mask_xr, pipelines_normed