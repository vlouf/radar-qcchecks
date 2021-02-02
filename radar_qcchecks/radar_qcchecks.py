import os
import traceback

import pyodim
import numpy as np
import pandas as pd
import dask
import dask.bag as db


def smooth_and_trim(x, window_len=11, window='hanning'):
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
    valid_windows = ['flat', 'hanning', 'hamming', 'bartlett', 'blackman',
                     'sg_smooth']
    if window not in valid_windows:
        raise ValueError("Window is on of " + ' '.join(valid_windows))

    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]

    if window == 'flat':  # moving average
        w = np.ones(int(window_len), 'd')
    elif window == 'sg_smooth':
        w = np.array([0.1, .25, .3, .25, .1])
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')

    return y[int(window_len / 2):len(x) + int(window_len / 2)]


def det_sys_phase(rhv, phidp, rhv_lev=0.6):
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
    def zdr_stats(zdrp):
        szdr = zdrp.std()
        aad = np.sum(np.abs(zdrp - zdrp.mean())) / len(zdrp)
        return szdr, aad

    sigma_phi = np.zeros_like(phidp)
    for i in range(phidp.shape[0]):
        sigma_phi[i, :] = pd.Series(np.ma.masked_where(~pos, phidp)[i, :]).rolling(15).std()

    sig_kdp = np.nanmedian(np.sqrt(3) * sigma_phi / (15 ** 1.5 * 0.25))

    szdr, aad = zdr_stats(zdr[pos])

    statistics = {
        "N": pos.sum(),
        "ZH_med": np.median(dbzh[pos]),
        "RHOHV_med": np.median(rhohv[pos]),
        "RHOHV_std": np.std(rhohv[pos]),
        "KDP_med": np.nanmedian(kdp[pos]),
        "KDP_std": np.nanstd(kdp[pos]),
        "ZDR_std": szdr,
        "ZDR_aad": aad,
        "PHIDP_med": np.nanmedian(sigma_phi),
        "PHIDP_std": np.nanstd(sigma_phi),
        "N_kdp_sample": (~np.isnan(sigma_phi)).sum(),
        "SIGMA_kdp": sig_kdp
    }

    return statistics


def read_data(infile):
    # Load radar
    radar = pyodim.read_odim(infile, True)

    # Read elevation
    radar[0] = radar[0].compute()

    # Load data
    dbzh = radar[0].DBZH.values
    zdr = radar[0].ZDR.values
    rhohv = radar[0].RHOHV.values
    phidp = radar[0].PHIDP.values
    kdp = radar[0].KDP.values

    dtime = pd.Timestamp(radar[0].time[0].values)

    # Offset to 0 phidp
    phasemin = det_sys_phase(rhohv, phidp)
    if np.isnan(phasemin):
        phasemin = np.median(phidp[~np.isnan(dbzh)])
    phidp = phidp - phasemin

    pos_lowcut = (
        ~np.isnan(dbzh) &
        (dbzh >= 20) & (dbzh <= 28)
    )

    if np.sum(pos_lowcut) < 100:
        return None
    if np.median(rhohv[pos_lowcut]) < 0.7:
        return None

    lowstats = get_statistics(dbzh, phidp, zdr, rhohv, kdp, pos_lowcut)
    lowstats.update({"time": dtime})

    return lowstats