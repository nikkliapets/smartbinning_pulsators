import numpy as np
import pandas as pd
import csv
import gc

import os
import shutil
from pathlib import Path

from scipy.ndimage import gaussian_filter1d
from scipy.stats import skew, kurtosis

from astropy.io import fits
from astropy.timeseries import LombScargle
import lightkurve as lk

from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

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

def get_frequencies_periods(smoothed_time, smoothed_flux):
    '''
    A helper function that extracts frequencies and periods from a light curve.
    Returns freuquencies, periods, and their respective amplitudes.
    '''
    dt = np.median(np.diff(smoothed_time))
    f_nyquist = 0.5 / dt

    f_min = 1e-4
    f_max = min(24, f_nyquist)

    frequency, power = LombScargle(smoothed_time, smoothed_flux).autopower(
        minimum_frequency=f_min,
        maximum_frequency=f_max,
        normalization='standard'
    )

    amplitude = np.sqrt(4 * power)

    period = 1 / frequency

    period = period[::-1]
    amplitude_p = amplitude[::-1]

    return frequency, amplitude, period, amplitude_p

# PART 2: FEATURE EXTRACTION

frequency_ranges = [
    (1e-4, 2.492680313306303),
    (2.492680313306303, 3.960646918540361),
    (3.960646918540361, 5.164376977515665),
    (5.164376977515665, 6.126573597493549),
    (6.126573597493549, 6.94479045337265),
    (6.94479045337265, 7.66938044231478),
    (7.66938044231478, 8.3669210762993),
    (8.3669210762993, 9.072344817545153),
    (9.072344817545153, 9.725128345005528),
    (9.725128345005528, 10.39721904902322),
    (10.39721904902322, 11.098065226686325),
    (11.098065226686325, 11.964778927386657),
    (11.964778927386657, 13.124580073719128),
    (13.124580073719128, 14.684744036122375),
    (14.684744036122375, 25)
]

period_ranges = [
    (0.04, 0.06556060081091043),
    (0.06556060081091043, 0.07046193144120899),
    (0.07046193144120899, 0.07620855141909326),
    (0.07620855141909326, 0.08227760044231479),
    (0.08227760044231479, 0.08982521931441209),
    (0.08982521931441209, 0.09945699594544784),
    (0.09945699594544784, 0.11143791743457428),
    (0.11143791743457428, 0.12800098415038702),
    (0.12800098415038702, 0.14952344268337633),
    (0.14952344268337633, 0.17843195724290453),
    (0.17843195724290453, 0.2198957537781054),
    (0.2198957537781054, 0.28078422410615556),
    (0.28078422410615556, 0.43106964246221896),
    (0.43106964246221896, 1.2639203317360854),
    (1.2639203317360854, 15.456813612237376)
]

def extract_features(fits_file, periodogram_dir=None):
    '''
    The main function that extracts statistical features from a FITS file's periodogram across defined frequency and period ranges.
    Returns a DataFrame with results.
    '''
    try:
        time, sap_flux = get_data(fits_file, pipeline='TGLC', ap=True)
        frequency, power, period, power_p = get_frequencies_periods(time, sap_flux)

        result = {}

        for i, (fmin, fmax) in enumerate(frequency_ranges):
            mask = (frequency >= fmin) & (frequency <= fmax)
            power_slice = power[mask]

            if len(power_slice) == 0:
                result[f'Amplitude{i+1}'] = 0
                result[f'Mean{i+1}'] = 0
                result[f'Variance{i+1}'] = 0
                result[f'Skew{i+1}'] = 0
                result[f'Kurt{i+1}'] = 0
            else:
                result[f'Amplitude{i+1}'] = np.max(power_slice)
                result[f'Mean{i+1}'] = np.mean(power_slice)
                result[f'Variance{i+1}'] = np.var(power_slice)
                result[f'Skew{i+1}'] = skew(power_slice)
                result[f'Kurt{i+1}'] = kurtosis(power_slice)

        for i, (pmin, pmax) in enumerate(period_ranges):
            mask = (period >= pmin) & (period <= pmax)
            power_slice = power_p[mask]

            if len(power_slice) == 0:
                result[f'Amplitude{i+1}p'] = 0
                result[f'Mean{i+1}p'] = 0
                result[f'Variance{i+1}p'] = 0
                result[f'Skew{i+1}p'] = 0
                result[f'Kurt{i+1}p'] = 0
            else:
                result[f'Amplitude{i+1}p'] = np.max(power_slice)
                result[f'Mean{i+1}p'] = np.mean(power_slice)
                result[f'Variance{i+1}p'] = np.var(power_slice)
                result[f'Skew{i+1}p'] = skew(power_slice)
                result[f'Kurt{i+1}p'] = kurtosis(power_slice)

        # Metadata
        parts = os.path.basename(fits_file).split('_')
        gaia_part = [p for p in parts if 'gaiaid' in p][0]
        gaia_id = gaia_part.split('-')[1]
        sector_part = gaia_part.split('-')[2]
        sector = str(int(sector_part[1:]))

        result['DR3'] = gaia_id
        result['Sector'] = sector
        result['FilePath'] = str(fits_file)

        if periodogram_dir is not None:
            os.makedirs(periodogram_dir, exist_ok=True)
            out_file = os.path.join(periodogram_dir, f'{gaia_id}_{sector}.npz')
            np.savez_compressed(out_file, frequency=frequency, power=power)

        return pd.DataFrame([result])

    except (TypeError, IndexError, OSError, ValueError):
        return None

# PART 3: PARALLEL PROCESSING

def process_file_wrapper(fits_file, periodogram_dir=None):
    '''
    A helper function that raps extract_features.
    '''
    try:
        df = extract_features(fits_file, periodogram_dir=periodogram_dir)
        return df if df is not None else None
    except Exception as e:
        print(f'Error processing {fits_file}: {e}')
        return None

def find_fits_in_subtree(subtree):
    '''
    A helper function that walks a subtree and collects all .fits files.
    '''
    matches = []
    for root, _, files in os.walk(subtree):
        matches.extend(
            Path(root) / f for f in files if f.lower().endswith('.fits')
        )
    return matches

def parallel_find_fits(master_folder, max_workers=32):
    '''
    A helper function that scans top-level subdirectories in parallel for .fits files.
    '''
    master_folder = Path(master_folder)
    top_subdirs = [p for p in master_folder.iterdir() if p.is_dir()]

    if not top_subdirs:
        top_subdirs = [master_folder]

    all_fits = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(find_fits_in_subtree, d): d for d in top_subdirs}
        for future in as_completed(futures):
            all_fits.extend(future.result())

    print(f'Discovered {len(all_fits):,} FITS files across {len(top_subdirs)} top-level folders.')
    return all_fits

def process_all_fits(master_folder, output_dir, n_jobs=100, max_discovery_workers=32, batch_size=100000):
    '''
    The main function that finds and processes all .fits files in parallel from a deep folder structure in chunks.
    '''
    master_folder = Path(master_folder)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_dir = output_dir / 'features'
    feature_dir.mkdir(parents=True, exist_ok=True)

    periodogram_dir = master_folder / 'pd'
    periodogram_dir.mkdir(parents=True, exist_ok=True)

    fits_files = parallel_find_fits(master_folder, max_workers=max_discovery_workers)

    total = len(fits_files)
    print(f'Processing {total:,} FITS files in chunks of {batch_size}...')

    for i in range(0, total, batch_size):
        batch_files = fits_files[i:i + batch_size]
        print(f'\nStarting batch {i // batch_size + 1}: {len(batch_files)} files...')

        results = Parallel(n_jobs=n_jobs)(
            delayed(process_file_wrapper)(file, periodogram_dir=periodogram_dir)
            for file in batch_files
        )

        valid_dfs = [df for df in results if df is not None]
        if not valid_dfs:
            print(f'No valid DataFrames produced in batch {i // batch_size + 1}. Skipping save.')
            continue

        batch_df = pd.concat(valid_dfs, ignore_index=True)

        batch_file = feature_dir / f'{master_folder.name}_features_batch{i // batch_size + 1}.parquet'
        batch_df.to_parquet(batch_file, index=False)
        print(f'Saved batch {i // batch_size + 1} to {batch_file}')

        del results, valid_dfs, batch_df
        gc.collect()

#sector_folders = [ADD THE SECTOR FOLDERS YOU WANT TO WORK WITH]
#base_path = 'ADD YOUR BASE BATH'

for sector in sector_folders:
    folder_path = os.path.join(base_path, sector)
    print(f'Processing {folder_path}...')

    process_all_fits(
        master_folder=folder_path,
        output_dir=base_path,
        n_jobs=100
    )

    gc.collect()
    print(f'Finished processing {sector}.')
    time.sleep(600)
