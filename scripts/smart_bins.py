import numpy as np
import pandas as pd
import csv

import os
import shutil
import glob
from pathlib import Path

from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from astropy.io import fits
from astropy.timeseries import LombScargle
import lightkurve as lk
import feets

from joblib import Parallel, delayed

# PART 1: DATA HANDLING

def smooth_time_series(time_series, sigma):
    '''
    A helper function to do Gaussian smoothing with a filter.
    '''
    smoothed_series = gaussian_filter1d(time_series, sigma=sigma, mode='nearest')
    return smoothed_series

def apply_quality_mask(time, flux, quality_mask, nan_threshold=10):
    '''
    A helper function to apply a quality mask, remove NaN values, and check for sufficient data.
    '''
    good_quality = quality_mask == 0
    time = time[good_quality]
    flux = flux[good_quality]
    
    mask = ~np.isnan(flux)
    if np.sum(mask) < nan_threshold:
        raise ValueError(f'The light curve contains too many NaN values.')
    return time[mask], flux[mask]

def remove_outliers(time, flux, sigma_threshold):
    '''
    A helper function to remove outliers from flux based on a sigma threshold.
    '''
    flux_mean = np.mean(flux)
    flux_std = np.std(flux)
    mask = np.abs(flux - flux_mean) < sigma_threshold * flux_std
    return time[mask], flux[mask]

def process_light_curve(time, flux, sigma_threshold, detrend_sigma):
    '''
    A helper function to processes the light curve by detrending with a Gaussian filter.
    '''
    time, flux = remove_outliers(time, flux, sigma_threshold)
    if time is None or flux is None:
        raise ValueError(f'The light curve {file} is empty or invalid.')
    smoothed = smooth_time_series(flux, sigma=detrend_sigma)
    detrended_flux = flux[:len(smoothed)] - smoothed
    detrended_time = time[:len(detrended_flux)]
    return detrended_time, detrended_flux + np.mean(flux)

def get_data(file, pipeline, ap=False):
    '''
    A helper function that opens a FITS file and detrends a light curve.
    Returns detrended flux and corresponding time points.
    '''
    if not os.path.exists(file):
        raise FileNotFoundError(f'The file {file} does not exist.')
    
    if pipeline == 'QLP':
        lc = lk.io.qlp.read_qlp_lightcurve(file)
        if lc is None:
            raise ValueError(f'Failed to read the light curve data from the file {file}.')
        time, flux = apply_quality_mask(
            np.array(lc['time'].value),
            np.array(lc['sap_flux'].value),
            lc.quality
        )
        if time is None or flux is None:
            raise ValueError(f'The light curve data for pipeline {pipeline} is empty or invalid.')
        return process_light_curve(time, flux, sigma_threshold=10, detrend_sigma=61)

    elif pipeline == 'TGLC':
        with fits.open(file) as hdul:
            if 'TESS_flags' not in hdul[1].data.columns.names or 'TGLC_flags' not in hdul[1].data.columns.names:
                raise ValueError(f'The FITS file {file} does not contain required columns 'TESS_flags' or 'TGLC_flags'.')
            flags = np.logical_and(
                hdul[1].data['TESS_flags'] == 0,
                hdul[1].data['TGLC_flags'] == 0
            )
            time = np.array(hdul[1].data['time'][flags])
            if ap == True:
                flux = np.array(hdul[1].data['cal_aper_flux'][flags])
            else:
                flux = np.array(hdul[1].data['cal_psf_flux'][flags])
            
            time, flux = apply_quality_mask(time, flux, quality_mask=np.zeros(len(time)))
            if time is None or flux is None:
                raise ValueError(f'The light curve data for pipeline {pipeline} is empty or invalid.')
            return process_light_curve(time, flux, sigma_threshold=3, detrend_sigma=100)

    else:
        raise ValueError(
            f'Pipeline {pipeline} is not supported. Supported pipelines: QLP, TGLC'
        )

# PART 2: BINS

def get_frequencies(smoothed_time, smoothed_flux, extended=False):
    '''
    A helper function to get frequencies.
    '''
    dt = np.median(np.diff(smoothed_time))
    f_nyquist = 0.5 / dt

    f_min = 1e-4
    f_max = min(70, f_nyquist)

    frequency, power = LombScargle(smoothed_time, smoothed_flux).autopower(
        minimum_frequency=f_min,
        maximum_frequency=f_max
    )

    return frequency, power

def get_ranges(frequency, power, bins, scaling):
    '''
    A helper function to get bin ranges for an individual source.
    '''
    peaks, properties = find_peaks(power, height=np.mean(power))
    peak_heights = properties['peak_heights']
    
    num_bins = 50 
    hist, bin_edges = np.histogram(frequency[peaks], bins=num_bins, weights=peak_heights)
    
    cumulative_power = np.cumsum(hist)
    normalized_cumulative = cumulative_power / np.max(cumulative_power)
    scaled_cumulative = np.power(normalized_cumulative, scaling)
    
    total_bins = bins 
    bin_indices = np.linspace(0, 1, total_bins + 1) 
    bin_edges_final = np.interp(bin_indices, scaled_cumulative, bin_edges[:-1])
    
    ranges = [(bin_edges_final[i], bin_edges_final[i + 1]) for i in range(len(bin_edges_final) - 1)]

    return ranges
    
def process_fits_from_df(df_files, pipeline, bins=15, scaling=1, ap=False):
    '''
    A helper function to process .fits files, for which paths are in the FilePath column of the dataframe.
    '''
    data = []
    col_names = [f'Range_{i+1}' for i in range(bins)]
    
    for file in df_files['FilePath']:
        try:
            time, flux = get_data(file, pipeline, ap=ap)
            frequency, power = get_frequencies(time, flux)
            ranges = get_ranges(frequency, power, bins, scaling)
            flat_ranges = [f'{r[0]:.5f}-{r[1]:.5f}' for r in ranges]
            row = {'file': file}
            row.update({col_names[i]: flat_ranges[i] for i in range(len(flat_ranges))})
            data.append(row)
        except Exception as e:
            print(f'Failed to process {file}: {e}')
    
    df = pd.DataFrame(data)
    return df

#dataframe = pd.read_csv(ADD YOUR CSV FILE WITH PATHS)
pipeline = 'TGLC'
df = process_fits_from_df(dataframe, pipeline, ap=True)

def average_range_bounds(df):
    '''
    The main function to get average ranges for a set of objects.
    '''
    avg_ranges = {}

    for col in df.columns:
        if col.startswith('Range_'):
            starts = []
            ends = []
            for val in df[col].dropna():
                if isinstance(val, str) and '-' in val:
                    try:
                        val_clean = val.strip().strip("'\"")
                        start_str, end_str = val_clean.split('-', 1)
                        start = float(start_str.strip("'\""))
                        end = float(end_str.strip("'\""))
                        starts.append(start)
                        ends.append(end)
                    except Exception as e:
                        continue
            
            if starts and ends:
                starts_clean = [s for s in starts if not pd.isna(s)]
                ends_clean = [e for e in ends if not pd.isna(e)]
                
                if starts_clean and ends_clean:
                    avg_start = sum(starts_clean) / len(starts_clean)
                    avg_end = sum(ends_clean) / len(ends_clean)
                    avg_ranges[col] = (avg_start, avg_end)

    return avg_ranges

avg = average_range_bounds(df)

df_avg_ranges = pd.DataFrame.from_dict(avg, orient='index', columns=['Start', 'End'])

df_avg_ranges = df_avg_ranges.reset_index().rename(columns={'index': 'Range'})

df_avg_ranges.to_csv('average_ranges.csv', index=False)
