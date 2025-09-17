# -*- coding: utf-8 -*-
"""
**ECG Signal Analysis With the Gaussian Multi-peak Fitting**

- Author: Liu Jun
- Last Update: 2025/03/30
"""
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


@dataclass
class Echo:
    """
    **Data class**

    Attributes:
    ----------
    ``index``:
    ``signal``:
    ``param0``:
    """
    index: np.ndarray | list
    signal: np.ndarray | list
    param0: np.ndarray | list = None
    bounds: tuple[list | np.ndarray, list | np.ndarray] = None
    param: np.ndarray | list = None
    gauss: np.ndarray | list = None


def multiGaussianes(x, *params):
    if len(params) % 3 != 1:
        raise ValueError(
            "The number of parameters does not match the specification, the parameter format [A_1, mu_1, sigma_1,"
            "... ,A_n, mu_n, sigma_n, B]")
    amps = params[::3]
    means = params[1::3]
    sigmas = params[2::3]
    return sum(amp * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2)) for mean, sigma, amp in zip(means, sigmas, amps)) + \
        params[-1]



def creat_echoes(signal, clusters):
    Echoes = {}
    for cluster_id in np.unique(clusters):
        index = np.where(clusters == cluster_id)[0]
        Echoes[cluster_id] = Echo(index, signal[index])
    return Echoes


def estimate_sigma_fwhm(signal, peak_position):
    peak_value = signal[peak_position]
    half_max = peak_value / 2

    left = peak_position
    while left > 0 and signal[left] > half_max:
        left -= 1
    right = peak_position
    while right < len(signal) - 1 and signal[right] > half_max:
        right += 1

    fwhm = 2 * max(peak_position - left, right - peak_position)  # Avoid the truncation bias
    sigma = fwhm / 2.3548
    return sigma


def find_positiones(signal, min_peak_height=0, min_peak_distance=5):
    # FIXME: QS is misplaced
    signal = signal - np.min(signal)
    peaks, properties = find_peaks(
        signal,
        height=signal.max() * min_peak_height,
        distance=min_peak_distance,
        prominence=0.001 * signal.max()
    )

    if len(peaks) < 3:
        raise ValueError(
            "The number of detected waveforms is less than 3, please adjust the parameters or check the signal")

    peak_prominences = properties['prominences']
    main_peaks = peaks[np.argsort(-peak_prominences)[:3]]
    p, r, t = np.sort(main_peaks)

    def find_valley(start, end):
        segment = signal[start:end + 1]
        valleys, _ = find_peaks(-segment)
        if len(valleys) == 0:
            return start + np.argmin(segment)
        return start + valleys[np.argmin(segment[valleys])]

    q = find_valley(p, r)
    s = find_valley(r, t)

    return p, q, r, s, t


class GaussianMultiPeakFitting:
    def __init__(self, signal: np.ndarray | list, clusters: np.ndarray | list,
                 positions: pd.DataFrame = None):
        self.echoes = creat_echoes(signal, clusters)
        self.positions: pd.DataFrame = positions

    def fit(self):
        params = pd.DataFrame([],
                              columns=['A_p', 'mu_p', 'sigma_p',
                                       'A_q', 'mu_q', 'sigma_q',
                                       'A_r', 'mu_r', 'sigma_r',
                                       'A_s', 'mu_s', 'sigma_s',
                                       'A_t', 'mu_t', 'sigma_t', 'B', 'MSE', 'PSNR', 'PCC'])
        error_list = pd.DataFrame([], columns=['index', 'error'])
        for echo_id in self.echoes.keys():
            try:
                # signal = self.echoes[echo_id].signal
                param, e, error_list = self.estimate_param0(echo_id)
                params.loc[echo_id] = np.append(param, e)

            except ValueError:
                continue
        return params, error_list

    def estimate_param0(self, i):
        signal = self.echoes[i].signal
        if self.positions is None:
            p, q, r, s, t = find_positiones(signal)
        else:
            p, q, r, s, t = self.positions.iloc[i - 1] - self.echoes[i].index[0]
            if np.any(np.isnan([p, q, r, s, t])):
                raise ValueError
        sigmas = np.zeros(5)
        temp_signal = signal - np.median(signal)
        for i, item in enumerate((p, q, r, s, t)):
            sigmas[i] = estimate_sigma_fwhm((-1) ** i * temp_signal, item)

        param0 = np.array([temp_signal[p], p, sigmas[0],
                           temp_signal[q], q, sigmas[1],
                           temp_signal[r], r, sigmas[2],
                           temp_signal[s], s, sigmas[3],
                           temp_signal[t], t, sigmas[4],
                           np.median(signal)])
        self.echoes[i].param0 = param0
        lower = np.array([temp_signal[p] - 0.5, p - 5, sigmas[0] * 0.5,
                          temp_signal[q] - 0.5, q - 5, sigmas[1] * 0.5,
                          temp_signal[r] - 0.5, r - 1, sigmas[2] * 0.5,
                          temp_signal[s] - 0.5, s - 5, sigmas[3] * 0.5,
                          temp_signal[t] - 0.5, t - 5, sigmas[4] * 0.5,
                          np.median(signal) - 0.5])
        upper = np.array([temp_signal[p] + 0.5, p + 5, sigmas[0] * 1.5,
                          0, q + 5, sigmas[1] * 1.5,
                          temp_signal[r] + 0.5, r + 1, sigmas[2] * 1.5,
                          0, s + 5, sigmas[3] * 1.2,
                          temp_signal[t] + 0.5, t + 5, sigmas[4] * 1.5,
                          np.median(signal) + 0.5])
        bounds = (lower, upper)
        time = np.arange(len(temp_signal))
        param = curve_fit(multiGaussianes, xdata=time, ydata=temp_signal, p0=param0, bounds=bounds, maxfev=5000, )
        self.echoes[i].param = param[0]
        # plt.plot(time, temp_signal)
        # for waveType in [p, q, r, s, t]:
        #     plt.scatter(waveType, temp_signal[waveType])
        # for i in [0, 3, 6, 9, 12]:
        #     plt.plot(time, multiGaussianes(time, *np.append(param0[i:i + 3], (np.median(signal)))))
        # plt.plot(time, multiGaussianes(time, *param0))
        # plt.plot(time, multiGaussianes(time, *param[0]))
        # plt.show()
        fit_y = multiGaussianes(time, *param[0])
        error = fit_y - temp_signal
        # print(np.mean(error ** 2), np.log10(np.max(fit_y) ** 2 / np.mean(error ** 2)) * 10)
        pcc = np.corrcoef(fit_y, temp_signal)[0, 1]
        sig = pd.DataFrame({
            'index': time - r,
            'error': np.abs(error),
        })
        return param[0], np.array(
            [np.mean(error ** 2), np.log10(np.max(fit_y) ** 2 / np.mean(error ** 2)) * 10, pcc]), sig
