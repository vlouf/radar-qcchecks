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
@date: 03/02/2021

.. autosummary::
    :toctree: generated/

    smooth_and_trim
    det_sys_phase
    get_statistics
    qccheck_radar_odim
"""
import warnings

import pyodim
import numpy as np
import pandas as pd


def smooth_and_trim(x, window_len: int = 11, window: str = "hanning"):
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


def det_sys_phase(rhv, phidp, rhv_lev: float = 0.6):
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


def get_statistics(dbzh, phidp, zdr, rhohv, kdp, pos):
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
        szdr = np.nanstd(zdrp)
        aad = np.sum(np.abs(zdrp - np.nanmean(zdrp))) / len(zdrp)
        return szdr, aad

    sigma_phi = np.zeros_like(phidp)
    for i in range(phidp.shape[0]):
        sigma_phi[i, :] = pd.Series(np.ma.masked_where(~pos, phidp)[i, :]).rolling(15).std()

    sig_kdp = np.nanmedian(np.sqrt(3) * sigma_phi / (15 ** 1.5 * 0.25))

    szdr, aad = zdr_stats(zdr[pos])

    statistics = {
        "N": pos.sum(),
        "ZH_med": np.median(dbzh[pos]),
        "RHOHV_med": np.nanmedian(rhohv[pos]),
        "RHOHV_std": np.nanstd(rhohv[pos]),
        "KDP_med": np.nanmedian(kdp[pos]),
        "KDP_std": np.nanstd(kdp[pos]),
        "ZDR_std": szdr,
        "ZDR_aad": aad,
        "PHIDP_med": np.nanmedian(sigma_phi),
        "PHIDP_std": np.nanstd(sigma_phi),
        "N_kdp_sample": (~np.isnan(sigma_phi)).sum(),
        "SIGMA_kdp": sig_kdp,
    }

    return statistics


def qccheck_radar_odim(
    infile: str,
    dbz_name: str = "DBZH",
    zdr_name: str = "ZDR",
    rhohv_name: str = "RHOHV",
    phidp_name: str = "PHIDP",
    kdp_name: str = "KDP",
):
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

    Returns:
    ========
    lowstats: dict
        Quality checks statistics in light rain for DP variables.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Load radar
        radar = pyodim.read_odim(infile, True)

        # Read elevation
        radar[0] = radar[0].compute()

        # Load data
        try:
            dbzh = radar[0]["DBZH_CLEAN"].values
        except KeyError:
            dbzh = radar[0][dbz_name].values
        zdr = radar[0][zdr_name].values
        rhohv = radar[0][rhohv_name].values
        phidp = radar[0][phidp_name].values
        kdp = radar[0][kdp_name].values

        dtime = pd.Timestamp(radar[0].time[0].values)

        # Offset to 0 phidp
        phasemin = det_sys_phase(rhohv, phidp)
        if np.isnan(phasemin):
            phasemin = np.median(phidp[~np.isnan(dbzh)])
        phidp = phidp - phasemin

        pos_lowcut = ~np.isnan(dbzh) & (dbzh >= 20) & (dbzh <= 28) & (rhohv > 0.7)

        if np.sum(pos_lowcut) < 100:
            return None
        if np.median(rhohv[pos_lowcut]) < 0.7:
            return None

        lowstats = get_statistics(dbzh, phidp, zdr, rhohv, kdp, pos_lowcut)
        lowstats.update({"time": dtime})

    return lowstats
