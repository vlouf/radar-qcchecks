"""
Quality control check of dual-polarization variables following the paper:

Marks, D. A., Wolff, D. B., Carey, L. D., Tokay, A., Systems, S., & County, B.
(2011). Quality control and calibration of the dual-polarization radar at
Kwajalein, RMI. Journal of Atmospheric and Oceanic Technology, 28(2), 181–196.
https://doi.org/10.1175/2010JTECHA1462.1

@title: radar_qcchecks
@author: Valentin Louf <valentin.louf@bom.gov.au>
@institutions: Bureau of Meteorology
@creation: 02/02/2021
@date: 04/02/2021

.. autosummary::
    :toctree: generated/

    smooth_and_trim
    det_sys_phase
    get_statistics
    qccheck_radar_odim
"""
import os
import contextlib
import warnings
from typing import Dict

import pyodim
import numpy as np
import numpy.typing as npt
import pandas as pd


def smooth_and_trim(x: np.ndarray, window_len: int = 11, window: str = "hanning") -> np.ndarray:
    """
    Smooth data using a window with requested size.
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    Parameters
    ----------
    x : array
        The input signal.
    window_len : int, optional
        The dimension of the smoothing window; should be an odd integer.
    window : str
        The type of window from 'flat', 'hanning', 'hamming', 'bartlett',
        'blackman' or 'sg_smooth'. A flat window will produce a moving
        average smoothing.
    Returns
    -------
    y : array
        The smoothed signal with length equal to the input signal.
    """
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x
    valid_windows = ["flat", "hanning", "hamming", "bartlett", "blackman", "sg_smooth"]
    if window not in valid_windows:
        raise ValueError("Window is on of " + " ".join(valid_windows))

    s = np.r_[x[window_len - 1 : 0 : -1], x, x[-1:-window_len:-1]]

    if window == "flat":  # moving average
        w = np.ones(int(window_len), "d")
    elif window == "sg_smooth":
        w = np.array([0.1, 0.25, 0.3, 0.25, 0.1])
    else:
        w = eval("np." + window + "(window_len)")

    y = np.convolve(w / w.sum(), s, mode="valid")

    return y[int(window_len / 2) : len(x) + int(window_len / 2)]


def det_sys_phase(rhv: np.ndarray, phidp: np.ndarray, rhv_lev: float = 0.6) -> float:
    """ Determine the system phase, see :py:func:`det_sys_phase`. """
    good = False
    phases = []
    for radial in range(phidp.shape[0]):
        meteo = rhv[radial, :] > rhv_lev
        mpts = np.where(meteo)
        if len(mpts[0]) > 25:
            good = True
            msmth_phidp = smooth_and_trim(phidp[radial, mpts[0]], 9)
            phases.append(msmth_phidp[0:25].min())
    if not good:
        return None
    return np.median(phases)


def get_statistics(
    dbzh: np.ndarray, phidp: np.ndarray, zdr: np.ndarray, rhohv: np.ndarray, kdp: np.ndarray, pos: np.ndarray,
) -> Dict:
    """
    Perform Marks et al. (2011) series of statistics for dual-polarisation
    quality checks.

    Parameters:
    ===========
    dbzh: ndarray<azimuth, range>
        Array of reflectivity (lowest elevation scan)
    phidp: ndarray<azimuth, range>
        Array of differential phase (lowest elevation scan)
    zdr: ndarray<azimuth, range>
        Array of differential reflectivity (lowest elevation scan)
    rhohv: ndarray<azimuth, range>
        Array of cross correlation ratio (lowest elevation scan)
    kdp: ndarray<azimuth, range>
        Array of specific differential phase (lowest elevation scan)
    pos: ndarray<azimuth, range>
        Mask for light rain filtering.

    Returns:
    ========
    statistics: dict
        DP quality checks statistics for given scan.
    """

    def zdr_stats(zdrp):
        szdr = np.std(zdrp)
        aad = np.sum(np.abs(zdrp - np.mean(zdrp))) / len(zdrp)
        return szdr, aad

    # PHIDP rolling window.
    wsize = 15  # Window size for avg phidp.
    sigma_phi = np.zeros_like(phidp)
    for i in range(phidp.shape[0]):
        try:
            sigma_phi[i, :] = pd.Series(np.ma.masked_where(~pos, phidp)[i, :]).rolling(wsize).std()
        except Exception:
            continue
    sigma_phi = np.ma.masked_invalid(sigma_phi).filled(np.NaN)
    nsample_kdp = (~np.isnan(sigma_phi)).sum()
    if nsample_kdp == 0:
        sig_kdp = np.NaN
        sig_phi_med = np.NaN
        sig_phi_std = np.NaN
    else:
        sig_kdp = np.nanmedian(np.sqrt(3) * sigma_phi / (wsize ** 1.5 * 0.25))
        sig_phi_med = np.nanmedian(sigma_phi)
        sig_phi_std = np.nanstd(sigma_phi)

    # ZDR stats.
    szdr, aad = zdr_stats(zdr[pos])

    statistics = {
        "N": pos.sum(),
        "ZH_med": np.median(dbzh[pos]),
        "RHOHV_med": np.median(rhohv[pos]),
        "RHOHV_std": np.std(rhohv[pos]),
        "KDP_med": np.median(kdp[pos]),
        "KDP_std": np.std(kdp[pos]),
        "ZDR_std": szdr,
        "ZDR_aad": aad,
        "PHIDP_med": np.nanmedian(phidp[pos]),
        "PHIDP_std": np.nanstd(phidp[pos]),
        "N_kdp_sample": nsample_kdp,
        "SIGMA_PHIDP_med": sig_phi_med,
        "SIGMA_PHIDP_std": sig_phi_std,
        "SIGMA_kdp": sig_kdp,
    }

    for k, v in statistics.items():
        if type(v) == np.ma.core.MaskedConstant:
            statistics[k] = np.NaN

    return statistics


def qccheck_radar_odim(
    infile: str,
    dbz_name: str = "DBZH",
    zdr_name: str = "ZDR",
    rhohv_name: str = "RHOHV",
    phidp_name: str = "PHIDP",
    kdp_name: str = "KDP",
    nsweep: int = 0,
) -> Dict:
    """
    Quality control check of dual-polarization variables of ODIM h5 input file.

    Parameters:
    ===========
    infile: str
        Input radar file in ODIM H5 format.
    dbz_name: str
        Reflectivity field name.
    zdr_name: str
        ZDR field name.
    rhohv_name: str
        RHOHV field name.
    phidp_name: str
        PHIDP field name.
    kdp_name: str
        KDP field name.
    nsweep: int
        Radar sweep number (pyodim reoder sweep by increasing elevation)

    Returns:
    ========
    lowstats: dict
        Quality checks statistics in light rain for DP variables.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # This silences standard output from pyodim library
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            # Load radar
            nradar = pyodim.read_odim(infile, True)

        # Read elevation
        radar = nradar[nsweep].compute()

        # Load data
        try:
            dbzh = radar["DBZH_CLEAN"].values
        except KeyError:
            dbzh = radar[dbz_name].values
        zdr = np.ma.masked_invalid(radar[zdr_name].values)
        rhohv = np.ma.masked_invalid(radar[rhohv_name].values).filled(0)
        phidp = radar[phidp_name].values
        kdp = np.ma.masked_invalid(radar[kdp_name].values)
        x = radar.x.values / 1000
        y = radar.y.values / 1000
        r = np.sqrt(x ** 2 + y ** 2)

        dtime = pd.Timestamp(radar.time[0].values)

        # Offset to 0 phidp
        phasemin = det_sys_phase(rhohv, phidp)
        if np.isnan(phasemin):
            phasemin = np.nanmedian(phidp[~np.isnan(dbzh)])
        phidp = phidp - phasemin

        pos_lowcut = ~np.isnan(dbzh) & (dbzh >= 20) & (dbzh <= 28) & (rhohv > 0.7) & (r < 150)

        if np.sum(pos_lowcut) < 100:
            return None

        lowstats = get_statistics(dbzh, phidp, zdr, rhohv, kdp, pos_lowcut)
        lowstats.update({"time": dtime})

    return lowstats
