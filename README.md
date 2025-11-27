# smartbinning_pulsators
Codes for the [Kliapets et al (2025)](https://ui.adsabs.harvard.edu/abs/2025A%26A...703A.240K/abstract) paper on finding candidate hybrid pulsators in the TESS data:
1. smart_bins.py: binning of the periodogram;
2. features_nominal.py: extracting features for nominal mission TESS light curves;
3. features_extended.py: extracting features for extended mission TESS light curves;
4. frequencies.py: extracting frequency information for population analysis;
5. fundamental_params.py: computing global parameters for population analysis.

For classification, see repository of [Dr. Pablo Huijse](https://github.com/phuijse/bagging_pu).

Please note that due to a small error in the plotting script, Figures 1, 3, 8, 9, C2, and D1 have an error in the units displayed. The flux plotted on Figures 1, C2, and D2 is mean-restored (detrended absolute flux), hence it is unitless (by error displayed in ppt). Peaks of the Lomb Scargle periodograms on Figures 1, 3, 8, 9, C2, and D1 are in a unitless power measure (by error displayed in ppt). Should you need to use any of these figures, corrected versions are available in the corrected_plots folder.
