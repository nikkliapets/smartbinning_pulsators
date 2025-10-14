import numpy as np
import pandas as pd

import os
import glob
import re

from scipy.ndimage import gaussian_filter1d

from astropy.io import fits
from astropy.timeseries import LombScargle
import lightkurve as lk

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

# PART 2: EXTRACTING FREQUENCY INFORMATION

def extract_periodogram_features(path):
    '''
    The main functon to extract frequency information from the light curve.
    '''
    try:
        time, flux = get_data(path, 'TGLC', ap=True)
        N = len(flux)
        T = time[-1] - time[0]
        cadence = np.diff(time)
        sampling = 1 / np.median(cadence)
        nyquist = sampling / 2

        ls = LombScargle(time, flux)
        frequency, power = ls.autopower(maximum_frequency=nyquist)
        fap_threshold = ls.false_alarm_level(0.01)

        significant_mask = power > fap_threshold
        if not np.any(significant_mask):
            raise ValueError('No significant peaks above the 1% FAP level.')

        frequency = frequency[significant_mask]
        power = power[significant_mask]

        sorted_idx = np.argsort(power)[::-1]

        def safe_amp(i):
            return np.sqrt(power[sorted_idx[i]]) if i < len(sorted_idx) else np.nan

        def safe_freq(i):
            return frequency[sorted_idx[i]] if i < len(sorted_idx) else np.nan

        freq_max = safe_freq(0)
        amp_max = safe_amp(0)
        freq_2nd = safe_freq(1)
        amp_2nd = safe_amp(1)

        model_flux = ls.model(time, freq_max)
        residuals = flux - model_flux
        sigma_noise = np.std(residuals)
        amp_uncert = np.sqrt(2) / N * sigma_noise

        def freq_uncertainty(A):
            if np.isnan(A) or A == 0:
                return np.nan
            return (3 * np.sqrt(6) / np.pi) * (sigma_noise / (A * T * np.sqrt(N)))

        low_mask = frequency <= 4
        high_mask = frequency > 4

        proximity_threshold = 0.1

        if np.any(low_mask):
            freq_low = frequency[low_mask][np.argmax(power[low_mask])]
            amp_low = np.sqrt(np.max(power[low_mask]))
        else:
            freq_low = np.nan
            amp_low = np.nan

        if not np.isnan(freq_low):
            multipliers = np.arange(2, 15.5, 0.5)
            harmonics = multipliers * freq_low
            harmonic_mask = np.full_like(frequency, False, dtype=bool)
            for h in harmonics:
                harmonic_mask |= np.abs(frequency - h) < proximity_threshold
            valid_high_mask = high_mask & ~harmonic_mask
        else:
            valid_high_mask = high_mask

        if np.any(valid_high_mask):
            freq_high = frequency[valid_high_mask][np.argmax(power[valid_high_mask])]
            amp_high = np.sqrt(np.max(power[valid_high_mask]))
        else:
            freq_high = np.nan
            amp_high = np.nan

        return {
            'freq_max': freq_max,
            'amp_max': amp_max,
            'freq_2nd': freq_2nd,
            'amp_2nd': amp_2nd,
            'freq_low': freq_low,
            'amp_low': amp_low,
            'freq_high': freq_high,
            'amp_high': amp_high,
            'amp_max_err': amp_uncert,
            'freq_max_err': freq_uncertainty(amp_max),
            'amp_2nd_err': amp_uncert,
            'freq_2nd_err': freq_uncertainty(amp_2nd),
            'amp_low_err': amp_uncert,
            'freq_low_err': freq_uncertainty(amp_low),
            'amp_high_err': amp_uncert,
            'freq_high_err': freq_uncertainty(amp_high)
        }

    except Exception as e:
        print(f'Error processing {path}: {e}')
        return {
            k: np.nan for k in [
                'freq_max', 'amp_max', 'freq_2nd', 'amp_2nd',
                'freq_low', 'amp_low', 'freq_high', 'amp_high',
                'amp_max_err', 'freq_max_err',
                'amp_2nd_err', 'freq_2nd_err',
                'amp_low_err', 'freq_low_err',
                'amp_high_err', 'freq_high_err'
            ]
        }

def process_periodogram_features_parallel(df, n_jobs=-1):
    results = Parallel(n_jobs=n_jobs)(
        delayed(extract_periodogram_features)(path) for path in df['FilePath']
    )

    results_df = pd.DataFrame(results)

    base_df = df[['FilePath', 'DR3']].reset_index(drop=True)
    combined_df = pd.concat([base_df, results_df], axis=1)

    return combined_df

#df = pd.read_parquet(ADD PATH TO THE PARQUET)
df_feat = process_periodogram_features_parallel(df_ext)
df_feat.to_csv('frequencies.csv')