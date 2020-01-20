from scipy.optimize import curve_fit
import numpy as np
import logging

logger = logging.getLogger('fithalo.src.halo')


# functional form for the model RC for a pseudo-isothermal spherical DM halo
def halo(r, v_h, r_c):
    return v_h * np.sqrt(1 - ((r_c / r) * (np.arctan(r / r_c))))


# function used when fitting for halo & M/L
def total(X, *p):
    v_h, r_c, d_ml, b_ml = p
    r, vgas, vdisk, vbulge = X
    return np.sqrt((halo(r, v_h, r_c)**2.) + (vgas**2.)
                   + (d_ml * (vdisk**2.)) + (b_ml * (vbulge**2.)))


"""The following two functions make use of curve_fit from the scipy
optimize package which uses non-linear least-squares methods to fit a
specified function to data. For bound problems it uses the trf
(Trust Region Reflective) algorithm for optimization. It defaults to
Levenberg-Marquardt for unconstrained problems. In this case, bounds
are necessary to prevent NaNs in the sqrt."""


# FITTING FUNCTION FOR FIXED M/L
def fit_dm_halo_fixed_ml(x, y, yerr, p_init=[150, 7]):
    # TODO: refactor to combine with fit_dm_halo
    # doesn't work (OptimizeWarning) without limits: 0 < V_h, R_c < 1000
    popt, pcov = curve_fit(halo, x, y, p_init, sigma=yerr, bounds=(0, 1000))
    perr = np.sqrt(np.diag(pcov))
    chi_sq = np.sum(((halo(x, *popt) - y) / yerr)**2.)
    red_chi_sq = chi_sq / (len(x) - len(popt))
    return popt, perr, chi_sq, red_chi_sq


# FITS FOR M/L & HALO SIMULTANEOUSLY
# imposes constraint 0.3 <= M/L <= 0.8
def fit_dm_halo(X, y, yerr, p_init=[150, 7, 0.5, 0.5]):
    # TODO: refactor to combine with fit_dm_halo_fixed_ml
    popt, pcov = curve_fit(total, X, y, p_init, sigma=yerr,
                           bounds=([0., 0., 0.3, 0.3], [1000, 1000, 0.8, 0.8]))
    perr = np.sqrt(np.diag(pcov))
    chi_sq = np.sum(((total(X, *popt) - y) / yerr)**2.)
    red_chi_sq = chi_sq / (len(y) - len(popt))
    return popt, perr, chi_sq, red_chi_sq


# FEEDS USER SPECIFIED M/L TO fit_dm_halo_fixed_ml
def do_fixed_ml_fit(r, v):
    # TODO: refactor to combine with do_fit
    # stellar V(s) already scaled by M/L
    vbary = np.sqrt((v[2] ** 2.) + (v[3] ** 2.) + (v[4] ** 2.))
    vfit = np.sqrt((v[0] ** 2.) - (vbary ** 2.)).dropna()[1:]  # skip r=0
    if len(vfit) <= 2:
        logger.error('Too many V_bary points > V_rot! Decrease M/L(s) and try again.')
        return
    rfit = r.reindex(vfit.index)
    vfit_err = ((v[0] * v[1]) / np.sqrt((v[0] ** 2.) - (vbary**2.))).reindex(vfit.index)
    # vfit_err = np.ones(len(vfit))  # uncomment to NOT weigh fit by errors
    popt, perr, chi_sq, red_chi_sq = fit_dm_halo_fixed_ml(rfit, vfit, vfit_err)
    halo_fit_params = {'v_h': popt[0], 'v_h_err': perr[0], 'r_c': popt[1], 'r_c_err': perr[1],
                       'chi_sq': chi_sq, 'red_chi_sq': red_chi_sq}
    return halo_fit_params


# M/L AS FREE PARAMETER, FEEDS TO fit_dm_halo
def do_fit(r, v):
    # TODO: refactor to combine with do_fixed_ml_fit
    vfit = v[0][1:]  # skip r=0
    rfit = r.reindex(vfit.index)
    vfit_err = v[1].reindex(vfit.index)
    # vfit_err = np.ones(len(vfit))  # uncomment to NOT weigh fit by errors
    vgas = v[2].reindex(vfit.index)
    vdisk = v[3].reindex(vfit.index)
    vbulge = v[4].reindex(vfit.index)
    X = (rfit, vgas, vdisk, vbulge)
    popt, perr, chi_sq, red_chi_sq = fit_dm_halo(X, vfit, vfit_err)
    halo_fit_params = {'v_h': popt[0], 'v_h_err': perr[0], 'r_c': popt[1], 'r_c_err': perr[1],
                       'chi_sq': chi_sq, 'red_chi_sq': red_chi_sq, 'd_ml': popt[2], 'b_ml': popt[3]}
    if all(vdisk[1:] == 0):
        halo_fit_params['d_ml'] = 0.
    if all(vbulge[1:] == 0):
        halo_fit_params['b_ml'] = 0.
    return halo_fit_params