import pandas as pd
import numpy as np

from concurrent.futures import ProcessPoolExecutor
from astropy.io.votable import parse_single_table

import gdr3bcg.bcg as bcg

Mbol_sun = 4.74
T_sun = 5772

def init_bcg():
    '''
    A helper function for bolometry.
    '''
    return bcg.BolometryTable()

def get_BC(row, tab):
    '''
    A helper function to compute bolometric correction.
    '''
    bc = tab.computeBc([row.teff_gspphot, row.logg_gspphot, row.mh_gspphot, 0])
    return bc[0] if isinstance(bc, (list, np.ndarray)) else float(bc)

def get_absmag(row):
    '''
    A helper function to compute absolute magnitude.
    '''
    return row.phot_g_mean_mag - 5.0 * np.log10(row.distance) + 5.0 - row.ag_gspphot

def compute_parameters(chunk):
    '''
    The main function to compute fundamental parameters.
    '''
    tab = init_bcg()
    
    chunk['phot_g_mean_mag_error'] = 1.0857 / chunk['phot_g_mean_flux_over_error']
    chunk['distance'] = 1000. / chunk['parallax']
    chunk['distance_error'] = 1000. * chunk['parallax_error'] / chunk['parallax']**2

    mask = chunk['distance_gspphot'].notna()
    chunk.loc[mask, 'distance'] = chunk.loc[mask, 'distance_gspphot']
    chunk.loc[mask, 'distance_error'] = 0.5 * (chunk.loc[mask, 'distance_gspphot_upper'] - chunk.loc[mask, 'distance_gspphot_lower'])

    chunk['ag_gspphot_error'] = 0.5 * (chunk['ag_gspphot_upper'] - chunk['ag_gspphot_lower'])
    chunk['phot_g_mean_mag_abs'] = chunk.apply(get_absmag, axis=1)
    chunk['abs_mag_error'] = np.sqrt(
        chunk['phot_g_mean_mag_error']**2 +
        (5 / np.log(10) * chunk['distance_error'] / chunk['distance'])**2 +
        chunk['ag_gspphot_error']**2
    )
    chunk['BC'] = chunk.apply(lambda row: get_BC(row, tab), axis=1)
    BC_error = 0.05

    chunk['Mbol'] = chunk['phot_g_mean_mag_abs'] + chunk['BC']
    chunk['Mbol_error'] = np.sqrt(chunk['abs_mag_error']**2 + BC_error**2)

    chunk['logL'] = 0.4 * (Mbol_sun - chunk['Mbol'])
    chunk['logL_err'] = 0.4 * chunk['Mbol_error']
    chunk['L'] = 10 ** chunk['logL']
    chunk['L_err'] = np.log(10) * chunk['L'] * chunk['logL_err']

    chunk['Teff'] = chunk['teff_gspphot']
    chunk['Teff_err'] = 0.5 * (chunk['teff_gspphot_upper'] - chunk['teff_gspphot_lower'])
    chunk['logTeff'] = np.log10(chunk['Teff'] / T_sun)
    chunk['logTeff_err'] = chunk['Teff_err'] / (chunk['Teff'] * np.log(10))

    chunk['logR'] = 0.5 * (chunk['logL'] - 4 * chunk['logTeff'])
    chunk['R'] = 10 ** chunk['logR']
    chunk['R_err'] = np.log(10) * chunk['R'] * np.sqrt(
        (0.5 * chunk['logL_err'])**2 + (4 * 0.5 * chunk['logTeff_err'])**2
    )
    chunk['logR_err'] = np.sqrt(
        (0.5 * chunk['logL_err'])**2 + (4 * 0.5 * chunk['logTeff_err'])**2
    )
    return chunk

def parallel_proc(df, n_jobs=48):
    '''
    A helper function to process entries in parallel.
    '''
    chunks = np.array_split(df, n_jobs)
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        results = list(executor.map(compute_parameters, chunks))
    return pd.concat(results)

#votable = parse_single_table(YOUR VOT TABLE, i.e. from Gaia)
df = votable.to_table().to_pandas()
df = parallel_proc(df)
df.to_csv('fundamental_params.csv')