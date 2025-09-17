# -*- coding: utf-8 -*-
"""
POT Method in the Streaming Data Detection

REFERENCE
---------

[1]. Siffer, A., Fouque, P.-A., Termier, A., Largouet, C., 2017. Anomaly detection in streams with extreme value theory,
in: Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD ’17.
Association for Computing Machinery, New York, NY, USA, pp. 1067–1075. https://doi.org/10.1145/3097983.3098144

[2]. Li, J., Di, S., Shen, Y., Chen, L., 2021. FluxEV: a fast and effective unsupervised framework for time-series anomaly
detection, in: Proceedings of the 14th ACM International Conference on Web Search and Data Mining. Presented at the
WSDM ’21: The Fourteenth ACM International Conference on Web Search and Data Mining, ACM, Virtual Event Israel, pp.
824–832. https://doi.org/10.1145/3437963.3441823
"""
import time

import matplotlib.pyplot as plt
import numpy as np
from typing import Literal
from scipy.optimize import minimize


def alarm(X: np.ndarray,
          q: float, quantile: float = 0.98,
          method: Literal['MOM', 'Grimshaw'] = 'Grimshaw',
          d: int = None
          ) -> np.ndarray:
    if d is None:
        _, zq = POT(X, q, quantile, method)
        return np.where(X - zq[:-1] > 0)[0]
    else:
        _, zq = DPOT(X, q, d, quantile, method)
        return np.where(X - zq[:-1] > 0)[0]


def POT(X: np.ndarray, q: float, quantile: float = 0.98, method: Literal['MOM', 'Grimshaw'] = 'Grimshaw') -> tuple[
    float, float]:
    """
    **Peak Over the Threshold Algotithm**
    :param X: input data
    :param q: risk level (probability of extreme value)
    :param quantile: high empirical quantile (defult: 0.98)
    :return: A tuple containing two elements (t, :math:`z_q`):
    :param method:
    """
    # Set initial threshold
    t = np.quantile(X, quantile)
    Yt = X[X > t] - t
    if method == 'Grimshaw':
        # Using Grimshaw's method to solve parameters gamma and sigma.
        gamma, sigma = _grimshaw(Yt)
    elif method == 'MOM':
        # Using MOM's method to solve parameters gamma and sigma. (Mothed of moments)
        gamma, sigma = _MOM(Yt)
    else:
        raise ValueError('method must be "MOM" or "Grimshaw"')

    # Calculate extreme value threshold.
    Nt = len(Yt)
    n = len(X)
    if gamma == 0:
        zq = t - sigma * np.log(q * n / Nt)
    else:
        zq = t + sigma / gamma * ((q * n / Nt) ** (-gamma) - 1)
    return t, zq


def DPOT(X: np.ndarray, q: float, d: int, quantile: float = 0.98, method: Literal['MOM', 'Grimshaw'] = 'Grimshaw') -> \
        tuple[np.ndarray, np.ndarray]:
    """
    **Peak Over the Threshold Algotithm with the Drift**
    :param X: input data
    :param q: risk level (probability of extreme value)
    :param d: window depth
    :param quantile: high empirical quantile (defult: 0.98)
    :param method
    :return: A tuple containing two elements (t, :math:`z_q`)
    """
    _X = np.zeros_like(X)
    Mi = np.empty(len(X) + 1, dtype=np.float64)
    # 利用窗口均值去除基线漂移
    cum_sum = np.concatenate([[0], np.cumsum(X, dtype=np.float64)])
    initial_mean = cum_sum[d] / d
    Mi[:d + 1] = initial_mean
    for i in range(d, len(X)):
        start = i - d + 1
        end = i
        window_sum = cum_sum[end + 1] - cum_sum[start]
        Mi[i + 1] = window_sum / d
    _X[d:] = X[d:] - Mi[d:len(X)]
    # 利用窗口中位数去除基线漂移
    # Mi[:d + 1] = np.full(d + 1, np.median(X[:d]))
    # for i in range(d, len(X)):
    #     _X[i] = X[i] - Mi[i]
    #     Mi[i + 1] = np.median(X[i - d + 1:i + 1])

    t, zq = POT(_X, q, quantile, method)
    return t + np.array(Mi), zq + np.array(Mi)


class SPOT:
    """
    **Streaming Peak Over the Threshold Algotithm**
    """

    def __init__(self, signal: np.ndarray, q: float, **kwargs):
        """
        **Initial(calibration) step**
        :param signal: Initial data
        :param q: Risk level (probability of extreme value)
        :param kwargs: Other keyword parameters for **POT Algotithm**. // Param 'quantile': high empirical quantile (defult: 0.98).
        """
        self.q = q
        self.t, zq = POT(signal, q, **kwargs)
        self.excesses: list = list(signal[signal > self.t] - self.t)
        self.zq: list[float] = list(np.full(len(signal) + 1, fill_value=zq))
        self.signal: list[float] = list(signal)
        self.n, self._inital_len = len(signal), len(signal)
        self.abnormal = {'index': [], 'value': []}

    def add(self, x: float) -> bool:
        """
        **Threshold update step**
        :param x: Input single data
        :return: Detect result, If True, mean the x is over the threshold :math:`z_q`.
        """
        self.signal.append(x)
        zq = self.zq[-1]
        if x > zq:
            self.abnormal['index'].append(len(self.signal) - 1)  # record the abnormal point index in all data.
            self.abnormal['value'].append(x)
            self.zq.append(zq)
            return True
        elif x > self.t:
            self.excesses.append(x - self.t)
            gamma, sigma = _grimshaw(np.array(self.excesses))
            Nt = len(self.excesses)
            self.n += 1
            if gamma == 0:
                zq = self.t - sigma * np.log(self.q * self.n / Nt)
            else:
                zq = self.t + sigma / gamma * ((self.q * self.n / Nt) ** (-gamma) - 1)
            self.zq.append(zq)
        else:
            self.n += 1
            self.zq.append(zq)
        return False

    def detect(self, data: np.ndarray | list[float]) -> dict[str, list[int | float]]:
        """
        **Detecting data anomalies as streaming**
        :param data: Data to be detected
        :return: A dictionary, contain the abnormal points index(key: index) and its value(key: value).
        """
        for i in data:
            self.add(i)
        return self.abnormal

    def view_abnormal(self, save_path: str = None):
        _plot_threshold(self.signal, self.zq, self.abnormal, self._inital_len, self.t, save_path)


class DSPOT:
    def __init__(self, signal: np.ndarray, d: int, q: float, **kwargs):
        """
        **Initial(calibration) step**
        :param signal: initial data
        :param d: Window depth
        :param q: Risk level (probability of extreme value)
        :param kwargs: Other parameters for **POT Algotithm**. // Param 'quantile': high empirical quantile (defult: 0.98).
        """
        self.q = q
        self.d = d
        _signal = np.zeros_like(signal)  # record the signal without drift baseline.
        self.Mi: list[float] = [0 for _ in range(d + 1)]  # drift baseline
        self.Mi[d] = np.mean(signal[:d])  # Local model with depth 't'
        for i in range(d, len(signal)):
            _signal[i] = signal[i] - self.Mi[i]
            self.Mi.append(signal[i - d + 1:i + 1].mean())
        self.W = list(signal[-d:])
        self.t, zq = POT(_signal[d:], q, **kwargs)
        self.excesses: list = list(_signal[_signal > self.t] - self.t)
        self._zq: list[float] = list(np.full(len(signal) + 1, fill_value=zq))  # 'z_q' without drift baseline.
        self.zq = list(np.array(self._zq) + np.array(self.Mi))  # real 'z_q'
        self.signal: list[float] = list(signal)
        self.n, self._inital_len = len(signal), len(signal)
        self.abnormal = {'index': [], 'value': []}

    def add(self, x: float) -> bool:
        """
        **Threshold update step**
        :param x: input single data
        :return: Detect result, If True, mean the x is over the threshold :math:`z_q`.
        """
        self.signal.append(x)
        zq = self._zq[-1]
        M_i = self.Mi[-1]
        _x = x - M_i
        if _x > zq:
            self.abnormal['index'].append(len(self.signal) - 1)  # record the abnormal point index in all data.
            self.abnormal['value'].append(x)
            self._zq.append(zq)
            self.Mi.append(M_i)
            self.zq.append(zq + M_i)
            return True
        elif _x > self.t:
            self.excesses.append(_x - self.t)
            gamma, sigma = _grimshaw(np.array(self.excesses))
            Nt = len(self.excesses)
            self.n += 1
            if gamma == 0:
                zq = self.t - sigma * np.log(self.q * self.n / Nt)
            else:
                zq = self.t + sigma / gamma * ((self.q * self.n / Nt) ** (-gamma) - 1)
            self._zq.append(zq)
            self.W = self.W[1:] + [x]
            self.Mi.append(np.mean(self.W))
        else:
            self.n += 1
            self._zq.append(zq)
            self.W = self.W[1:] + [x]
            self.Mi.append(np.mean(self.W))
        self.zq.append(zq + M_i)
        return False

    def detect(self, data: np.ndarray | list[float]) -> dict[str, list[int | float]]:
        """
        **Detecting data anomalies as streaming**
        :param data: Data to be detected
        :return: A dictionary, contain the abnormal points index(key: index) and its value(key: value)
        """
        for i in data:
            self.add(i)
        return self.abnormal

    def view_abnormal(self, save_path: str = None):
        _plot_threshold(self.signal, self.zq, self.abnormal, self._inital_len, self.t + np.array(self.Mi), save_path)


# ==========
# Internals
# ==========

def _MOM(y: np.ndarray):
    avg = np.mean(y)
    var = np.var(y)
    sigma = 0.5 * avg * (1 + avg ** 2 / var)
    gamma = 0.5 * (1 - avg ** 2 / var)
    return gamma, sigma


def _grimshaw(y: np.ndarray, epsilon: float = 1e-8, k: int = 10) -> (float, float):
    ymin = np.min(y)
    ymax = np.max(y)
    ymean = np.mean(y)
    Nt = len(y)

    def log_likelihood(gamma, sigma):
        result = np.zeros_like(gamma)
        for i, (g, s) in enumerate(zip(gamma, sigma)):
            if g != 0:
                result[i] = -Nt * np.log(s) - (1 + 1 / g) * np.sum(
                    np.log(1 + g / s * y))
            else:
                result[i] = Nt * (1 + np.log(np.mean(y)))
        return result

    def u(x):
        return (1 / (1 + x.reshape(-1, 1) * y.reshape(1, -1))).mean(axis=1)

    def v(x):
        return (np.log(1 + x.reshape(-1, 1) * y.reshape(1, -1))).mean(axis=1) + 1

    def w(x):
        return u(x) * v(x) - 1

    def optimization(x):
        return np.sum(w(x) ** 2)

    def gradient(x):
        jac_u = (y / (1 + x.reshape(-1, 1) * y.reshape(1, -1)) ** 2).mean(axis=1)
        jac_v = (y / (1 + x.reshape(-1, 1) * y.reshape(1, -1))).mean(axis=1)
        return jac_u * v(x) + u(x) * jac_v

    if epsilon >= 1 / (2 * ymax):
        epsilon = 1 / (ymax * (k + 1))

    bound1 = (-1 / ymax + epsilon, -epsilon)
    bound2 = (2 * (ymean - ymin) / (ymean * ymin), 2 * (ymean - ymin) / ymin ** 2)

    roots = set()
    for bounds in [bound1, bound2]:
        x0 = np.linspace(bounds[0], bounds[1], k, endpoint=True)
        root = minimize(optimization, x0, method='L-BFGS-B', jac=gradient, bounds=[bounds] * k).x
        roots.update(root)

    roots = np.array(list(roots))
    gammas = v(roots) - 1
    sigmas = gammas / roots
    lle = log_likelihood(gammas, sigmas)

    ii = np.argmax(lle)
    return gammas[ii], sigmas[ii]


def _plot_threshold(data: np.ndarray | list, zq: np.ndarray | list, abnormal: dict = None,
                    inital_len: int = None, t: np.ndarray | list | float = None, save: str = None) -> None:
    plt.figure(figsize=(16, 5))
    plt.plot(data)
    if inital_len is not None:
        if isinstance(t, float):
            plt.plot(np.full(inital_len, fill_value=t))
        if isinstance(t, (np.ndarray, list)):
            plt.plot(t[:inital_len], '--', label="$t$")
    plt.plot(zq, '-.', label="$z_q$")
    if abnormal is not None:
        plt.scatter(abnormal['index'], abnormal['value'], c='r', marker='x', label='Abnormal Point')
    plt.xlim(0, len(data))
    plt.tight_layout()
    plt.legend(loc='best')
    if save is not None:
        plt.savefig(save)
    else:
        plt.show()
