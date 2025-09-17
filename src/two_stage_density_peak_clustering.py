# -*- coding: utf-8 -*-
"""
Segment Heartbeats Based on the Density Peak Clustering Algorithm For the ECG Signals
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pywt
import tqdm
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from tqdm import trange

from GaussECG.peak_over_threshold import alarm

"""Picture format"""
FontSize: int = 18  # label font size
plt.rcParams['font.size'] = FontSize - 2
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams["savefig.transparent"] = True
plt.rcParams["savefig.dpi"] = 900
plt.rcParams["savefig.bbox"] = 'tight'


class TwoStageDPC:
    """
        **Two-stage Density Peak Clustering Algorithm for the ECG Signals**
        """

    @dataclass(kw_only=True)
    class Params:
        """聚类参数"""
        rho: NDArray
        delta: NDArray
        pointer: NDArray

    @dataclass(kw_only=True)
    class Wave:
        """
        **Dataclass for ECG feature points**
        """
        r_locs: NDArray
        p_locs: NDArray = field(init=False)
        q_locs: NDArray = field(init=False)
        s_locs: NDArray = field(init=False)
        t_locs: NDArray = field(init=False)

        def __post_init__(self):
            self.p_locs = np.full_like(self.r_locs, fill_value=np.nan, dtype=float)
            self.q_locs = np.full_like(self.r_locs, fill_value=np.nan, dtype=float)
            self.s_locs = np.full_like(self.r_locs, fill_value=np.nan, dtype=float)
            self.t_locs = np.full_like(self.r_locs, fill_value=np.nan, dtype=float)

        def add(self, i: int, r_loc: int | float, p_loc: int | float, q_loc: int | float, s_loc: int | float,
                t_loc: int | float):
            self.r_locs[i] = r_loc
            self.p_locs[i] = p_loc
            self.q_locs[i] = q_loc
            self.s_locs[i] = s_loc
            self.t_locs[i] = t_loc

        def to_df(self):
            return pd.DataFrame({
                'P': self.p_locs,
                'Q': self.q_locs,
                'R': self.r_locs,
                'S': self.s_locs,
                'T': self.t_locs,
            })

        def plot(self, signal, ax):
            """特征点位置可视化"""
            for item, c, masker, label in zip([self.r_locs, self.p_locs, self.q_locs, self.s_locs, self.t_locs],
                                              ['red', 'green', 'blue', 'yellow', 'orange'],
                                              ['v', '*', '+', '+', '*'],
                                              ['R', 'P', 'Q', 'S', 'T']):
                mask = ~np.isnan(item)
                valid_locs = item[mask].astype(int)
                valid_signals = signal[valid_locs]
                ax.scatter(valid_locs, valid_signals, s=15, c=c, label=label, marker=masker, zorder=2)
                ax.legend(ncol=5, loc='lower center', fontsize=FontSize - 6,
                          handletextpad=0.3, columnspacing=0.5, frameon=False,
                          bbox_to_anchor=(0.5, -0.23))

    def __init__(
            self,
            fs: float,
            win_len: int = 15,
            edge_len: int = 2,
            qr_max_time: float = 0.12,
            qrs_max_time: float = 0.18,
            pq_max_time: float = 0.30,
            st_seg_time: float = 0.15,
    ):
        """
        Initialising the detector.
        :param fs: The sampling frequency of the ECG signal (in Hz, samples/second).
        :param win_len: The window size for computing clustering parameters for large-scale data.
        :param edge_len: The window redundancy edge size.
        """
        self.fs = fs
        self._win_size: int = int(win_len * fs)
        self._edge: int = int(edge_len * fs)

        self.signal = None
        self.centers = None
        self.params: TwoStageDPC.Params | None = None
        self.data: TwoStageDPC.Wave | None = None

        self.rr = None
        self.cluster = None
        self.wave: TwoStageDPC.Wave | None = None

        # bound condition
        self.qr_max = int(qr_max_time * self.fs)
        self.qrs_max = int(qrs_max_time * self.fs)
        self.pq_max = int(pq_max_time * self.fs)
        self.st_seg = int(st_seg_time * self.fs)

    def run(
            self,
            signal: NDArray,
    ):
        ...

    def clc_params(
            self,
            signal: NDArray = None,
    ):
        if signal is not None:
            self.signal = signal
        elif self.signal is None:
            raise AttributeError("Attribute or paramster 'signal' not provided.")

        N = len(self.signal)
        rho = self._cal_local_density_Gaussian()

        if N > self._win_size * 10:
            delta = np.full(N, fill_value=-1, dtype=int)
            pointer = np.full(N, fill_value=-1, dtype=int)
            for i in trange(int(N / self._win_size), desc='Calculation parameters (in block)'):
                if i == 0:
                    _delta, _pointer = self._cal_distance(rho[:int(self._win_size + self._edge)])
                    delta[:self._win_size] = _delta[:self._win_size]
                    pointer[:self._win_size] = _pointer[:self._win_size]
                elif i == int(N / self._win_size) - 1:
                    _delta, _pointer = self._cal_distance(rho[int(i * self._win_size - self._edge):])
                    delta[i * self._win_size:] = _delta[self._edge:]
                    pointer[i * self._win_size:] = _pointer[self._edge:]
                else:
                    _delta, _pointer = self._cal_distance(
                        rho[int(i * self._win_size - self._edge):int((i + 1) * self._win_size + self._edge)])
                    delta[i * self._win_size:(i + 1) * self._win_size] = _delta[self._edge:-self._edge]
                    pointer[i * self._win_size:(i + 1) * self._win_size] = _pointer[self._edge:-self._edge]
        else:
            delta, pointer = self._cal_distance(rho)
        self.params = self.Params(rho=rho, delta=delta, pointer=pointer)

    def decide(
            self,
            alarm_delta: dict[str, int | float] = None,
            alarm_rho: dict[str, int | float] = None,
            show: bool = False,
    ):
        if self.params is None:
            raise AttributeError("Attribute or paramster 'params' not provided, please run clc_params() first.")
        if alarm_delta is None:
            alarm_delta: dict[str, int | float] = {'q': 0.04, 'd': int(self.fs), 'quantile': 0.95}
        if alarm_rho is None:
            alarm_rho: dict[str, int | float] = {'q': 0.01, 'd': int(self.fs), 'quantile': 0.95}
        self.centers = np.intersect1d(
            alarm(self.params.rho, method='MOM', **alarm_rho),
            alarm(self.params.delta, method='MOM', **alarm_delta)
        )
        if len(self.centers) == 0:
            raise ValueError("No centers detected.")
        rr = np.diff(self.centers)

        # _counter = 0
        # while len(rr) > 1 and any(rr > 2 * self.fs):
        #     idxs = np.where(rr > 2 * self.fs)[0]
        #     centers = self.centers
        #     for idx in idxs:
        #         _left, _right = int(self.centers[int(idx)] + 0.5 * self.fs), int(
        #             self.centers[int(idx + 1)] - 0.5 * self.fs)
        #         _centers = np.argmax(self.params.delta[_left:_right]) + _left
        #         centers = np.append(centers, int(_centers))
        #     self.centers = np.sort(centers)
        #     rr = np.diff(self.centers)
        #     _counter += 1
        #     if _counter > 10:
        #         break

        rr = np.insert(rr, 0, np.minimum(rr[0], 2 * self.centers[0]))
        rr = np.insert(rr, -1, np.minimum(rr[-1], 2 * self.centers[-1]))
        self.rr = rr

        if show:
            fig, ax = self.plot_decide()
            ax.axvline(float(np.quantile(self.params.rho, alarm_rho['quantile'])), color='r', linestyle='--')
            ax.axhline(float(np.quantile(self.params.delta, alarm_delta['quantile'])), color='r', linestyle='--')
            return fig, ax
        return None

    def adjust_pointer(self):
        if self.params is None:
            raise AttributeError("Attribute or paramster 'params' not provided, please run clc_params() first.")

        # Boundary Condition
        wave: TwoStageDPC.Wave = self.Wave(r_locs=self.centers)
        for i in tqdm.trange(len(self.centers), desc='Adjusting pointer'):
            l = self.centers[i] - min(self.centers[i], self.rr[i] // 2)
            r = self.centers[i] + self.rr[i + 1] // 2
            local_rho = self.params.rho[l:r]
            local_delta = self.params.delta[l:r]
            r_loc = min(self.centers[i], self.rr[i] // 2)  # the r peak location in the local signal

            # Condition 1: the distance between the Q-wave Peak and R-wave Peak less than 120ms.
            q_bound = max(r_loc - self.qr_max, 0)
            if q_bound == r_loc:
                continue
            q_loc = np.argmin(local_rho[q_bound:r_loc]) + q_bound

            # Condition 2: the distance between the S-wave Peak and Q-wave Peak less than 180ms.
            s_bound = q_loc + self.qrs_max
            if s_bound <= r_loc:
                continue
            s_loc = np.argmin(local_rho[r_loc:s_bound]) + r_loc

            # Condition 3: the distance between the P-wave Peak and Q-wave Peak less than 300ms.
            p_bound = q_loc - self.pq_max
            if p_bound > 0:
                p_loc = np.argmax(local_delta[p_bound:q_loc]) + p_bound
                self.params.pointer[p_loc + l] = self.centers[i]
            else:
                p_loc = np.nan

            # Condition 4: the distance between the T-wave Peak and S-wave Peak more than 150ms.
            base_line = np.median(local_rho)  # baseline for rho
            t_bound = s_loc + self.st_seg
            if t_bound >= len(local_rho):
                t_loc = np.nan
            else:
                t_loc = np.argmax(local_delta[t_bound:]) + t_bound
                self.params.pointer[t_loc + l] = self.centers[i]
                # t_temp1 = np.argmin(local_rho[t_bound:] - base_line) + t_bound  # maybe inverted T-wave
                # t_temp2 = np.argmax(local_delta[t_bound:]) + t_bound
                #
                # if local_rho[t_temp1] < -local_rho[t_temp2] + base_line * 2:
                #     t_loc = t_temp1
                #     self.params.pointer[np.where(local_rho[t_bound:] < base_line)[0] + t_bound + l] = self.centers[i]
                # else:
                #     t_loc = t_temp2
                #     self.params.pointer[t_loc + l] = self.centers[i]

            wave.add(i=i,
                     r_loc=r_loc + l, p_loc=p_loc + l, q_loc=q_loc + l, s_loc=s_loc + l, t_loc=t_loc + l)
        self.wave = wave

    def clustering(self):
        # sort_rho = np.sort(self.params.rho)
        sort_idx = np.argsort(-self.params.rho)
        cluster = np.zeros(len(self.signal), dtype=int)
        for c, peak in enumerate(self.centers):
            cluster[peak] = int(c + 1)
        for i in tqdm.tqdm(sort_idx[1:], desc='Clustering'):
            if cluster[i] == 0:
                cluster[i] = cluster[self.params.pointer[i]]
        self.cluster = cluster

    def plot_decide(
            self,
    ) -> tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots(1, 1, figsize=(9, 9))
        ax.scatter(self.params.rho, self.params.delta, s=10, edgecolors='none', facecolors='black', linewidths=1,
                   zorder=2)
        if self.centers.shape[0] != 0:
            rho_c, delta_c = self.params.rho[self.centers], self.params.delta[self.centers]
            ax.scatter(rho_c, delta_c, s=30, edgecolors='#ed2225', facecolors='none', linewidths=1, zorder=1)
        if np.max(self.params.delta) > np.mean(self.params.delta) * 10:
            ax.semilogy()
            ax.set_ylim(0.9, np.max(self.params.delta) + 100)
        else:
            ax.set_ylim(0, np.max(self.params.delta) * 1.05)
        fig.tight_layout()
        ax.set_ylabel('$delta$')
        ax.set_xlabel('$rho$')
        return fig, ax

    def plot_peak(self):
        fig, ax = plt.subplots(1, 1, figsize=(36, 4))
        ax.plot(self.signal, 'black')
        if self.wave is not None:
            self.wave.plot(signal=self.signal, ax=ax)
        elif self.centers is not None:
            ax.scatter(self.centers, self.signal[self.centers])
        ax.set_xlim(0, len(self.signal))
        fig.tight_layout()
        return fig, ax

    def plot_cluster(self):
        fig, ax = plt.subplots(1, 1, figsize=(36, 4))
        ax.plot(self.signal)
        for i in set(self.cluster) - {0, max(self.cluster)}:
            x = int(np.where(self.cluster == i)[0][-1])
            ax.axvline(x, color='red', linestyle='--')
        plt.show()

    def _cal_local_density_Gaussian(self, sigma: float = 2, radius: int = 2) -> NDArray:
        """Calculate of local density using the Gaussian Kernel weights"""
        rho = gaussian_filter1d(self.signal, sigma=sigma, radius=radius)
        return rho

    def _cal_local_density_Ricker(self) -> NDArray:
        """Calculate of local density using the Ricker wavelet (Mexican-hat wavle) weights"""
        # QRS complex duration about 0.06~0.10s
        qrs_width = int(0.1 * self.fs)
        scale = qrs_width / 5.0
        coef, _ = pywt.cwt(
            self.signal,
            [scale],
            'mexh',
            sampling_period=1.0 / self.fs
        )
        rho = coef[0]
        return rho

    @staticmethod
    def _cal_distance(rho: NDArray) -> tuple[NDArray, NDArray]:
        """Calculate minimum distance between the data point i and any other point with higher density"""
        N = len(rho)
        delta = np.full(N, fill_value=-1, dtype=int)
        pointer = np.full(N, fill_value=-1, dtype=int)
        sort_index = np.argsort(-rho)

        for i in range(1, N):
            idx = sort_index[i]
            higher_idx = sort_index[:i]
            distance = np.abs(higher_idx - idx)
            min_idx = np.argmin(distance)
            delta[idx] = distance[min_idx]
            pointer[idx] = higher_idx[min_idx]
        delta[sort_index[0]] = N
        return delta, pointer


if __name__ == '__main__':
    from Toolbox.DBtool import mitArr, luecgdb


    def test_real():
        db = mitArr()
        ecg = db.record('100')
        tsdpc = TwoStageDPC(ecg.fs)
        tsdpc.clc_params(ecg.signal)
        tsdpc.decide()
        tsdpc.adjust_pointer()
        tsdpc.plot_peak()
        tsdpc.clustering()
        tsdpc.plot_cluster()


    def test_real2():
        db = luecgdb()
        ecg = db.record('data/1')
        tsdpc = TwoStageDPC(ecg.fs)
        tsdpc.clc_params(ecg.signal)
        tsdpc.decide()
        tsdpc.adjust_pointer()
        tsdpc.plot_peak()
        tsdpc.clustering()
        tsdpc.plot_cluster()


    test_real()
    test_real2()
