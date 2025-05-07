import pandas as pd
import numpy as np
from timedomain import TimeDomain as td
from initialization import InitParameter as ip
from frequencydomain import FrequencyDomain as fd
from harmonic_sildband_table import HarmonicSildband as hs
from typing import Tuple, Any

ip = ip()

class FilterProcess:
    @staticmethod
    def NA4(amp: pd.DataFrame, m: int) -> Tuple[float, float, float]:
        lastDegree = amp['Degree'].iloc[-1]
        segment = m
        degreeSegment = np.arange(0, lastDegree + (lastDegree / segment), (lastDegree / segment))
        ampMean = amp['Acc'].agg(np.mean)
        allAmpCount = len(amp)
        totalSumSegment = 0

        for i in range(1, len(degreeSegment)):
            mask1 = amp['Degree'] > degreeSegment[i - 1]
            mask2 = amp['Degree'] <= degreeSegment[i]
            amp2 = amp[mask1 & mask2]
            amp2Mean = amp2['Acc'].agg(np.mean)
            amp3 = pd.DataFrame({'Degree': amp2['Degree'], 'Acc': amp2['Acc'], 'Mean': amp2Mean})
            sumSegment = sum((np.float64(amp3['Acc']) - np.float64(amp3['Mean'])) ** 2)
            totalSumSegment = sumSegment + totalSumSegment

        divisionTotalSumSegment = (totalSumSegment / segment) ** 2
        totalSumAll = sum((np.float64(amp['Acc']) - ampMean) ** 4) * allAmpCount
        na4 = totalSumAll / divisionTotalSumSegment if divisionTotalSumSegment != 0 else np.nan

        return na4, totalSumAll, divisionTotalSumSegment

    @staticmethod
    def NA4S(ifft_tsa: pd.DataFrame, good_ifft_tsa: pd.DataFrame, m: int) -> Tuple[float, float]:
        _, totalSumAll, _ = FilterProcess.NA4(ifft_tsa, m)
        _, _, goodSumSegment = FilterProcess.NA4(good_ifft_tsa, m)
        na4s = totalSumAll / goodSumSegment if goodSumSegment != 0 else np.nan
        return na4s, goodSumSegment

    @staticmethod
    def FM4(ifft_tsa: pd.Series) -> float:
        ifft_tsa_mean = ifft_tsa.agg(np.sum) / len(ifft_tsa)
        pdataCombine = pd.DataFrame({'ifft_tsa': ifft_tsa, 'mean': ifft_tsa_mean})
        difference = pdataCombine['ifft_tsa'] - pdataCombine['mean']
        denominator = np.sum(difference ** 2) ** 2
        fm4 = ((len(difference)) * np.sum(difference ** 4)) / denominator if denominator != 0 else np.nan
        return fm4

    @staticmethod
    def M6A(ifft_tsa: pd.Series) -> float:
        ifft_tsa_mean = ifft_tsa.agg(np.sum) / len(ifft_tsa)
        pdataCombine = pd.DataFrame({'ifft_tsa': ifft_tsa, 'mean': ifft_tsa_mean})
        difference = pdataCombine['ifft_tsa'] - pdataCombine['mean']
        denominator = np.sum(difference ** 2) ** 3
        f6a = (((len(difference)) ** 2) * np.sum(difference ** 6)) / denominator if denominator != 0 else np.nan
        return f6a

    @staticmethod
    def M8A(ifft_tsa: pd.Series) -> float:
        ifft_tsa_mean = ifft_tsa.agg(np.sum) / len(ifft_tsa)
        pdataCombine = pd.DataFrame({'ifft_tsa': ifft_tsa, 'mean': ifft_tsa_mean})
        difference = pdataCombine['ifft_tsa'] - pdataCombine['mean']
        denominator = np.sum(difference ** 2) ** 4
        m8a = (((len(difference)) ** 3) * np.sum(difference ** 8)) / denominator if denominator != 0 else np.nan
        return m8a

    @staticmethod
    def ER(amp: pd.Series, fs: int, fft: Any) -> float:
        _, sideband_ifft_tsa_output = FilterProcess.FilterSideband(amp, fs, fft)
        rms = td.rms(amp)
        sidebandRms = td.rms(sideband_ifft_tsa_output['Acc'])
        er = sidebandRms / rms if rms != 0 else np.nan
        return er

    @staticmethod
    def FilterHarmonic(amp: pd.Series, fs: int, fft: Any) -> Any:
        tsa_fftoutput, _, _, _ = fd.tsa_fft_fm0_slf(amp, fs, fft)
        _, max_harmonic_freq_combine = hs.Tsa_Harmonic(tsa_fftoutput)
        _, max_sideband_freq_combine = hs.Sildband(tsa_fftoutput)

        indexlist1 = tsa_fftoutput[tsa_fftoutput['multiply_freqs'].isin(max_harmonic_freq_combine['multiply_freqs'].tolist())].index.tolist()
        tsa_fftoutput = tsa_fftoutput.drop(indexlist1)
        indexlist2 = tsa_fftoutput[tsa_fftoutput['multiply_freqs'].isin(max_sideband_freq_combine['multiply_freqs'].tolist())].index.tolist()
        tsa_fftoutput = tsa_fftoutput.drop(indexlist2)

        ifft_tsa_output, time_value = fd.ifft_process(tsa_fftoutput['tsa_fft'])
        return ifft_tsa_output

    @staticmethod
    def FilterSideband(amp: pd.Series, fs: int, fft: Any) -> Tuple[Any, Any]:
        tsa_fftoutput, _, _, _ = fd.tsa_fft_fm0_slf(amp, fs, fft)
        _, max_harmonic_freq_combine = hs.Tsa_Harmonic(tsa_fftoutput)
        _, max_sideband_freq_combine = hs.Sildband(tsa_fftoutput)

        def get_main_freq(df, freq, rng):
            mask1 = df['multiply_freqs'] >= freq - rng
            mask2 = df['multiply_freqs'] <= freq + rng
            main = df[mask1 & mask2]
            main = df[df['tsa_abs_fft'] == np.max(main['tsa_abs_fft'])]
            return main.iloc[0:1], main.iloc[1:2]

        max_mortor_gear1, max_mortor_gear2 = get_main_freq(tsa_fftoutput, ip.mortor_gear, ip.side_band_range)
        max_belt_si1, max_belt_si2 = get_main_freq(tsa_fftoutput, ip.belt_si, ip.side_band_range)
        max_mortor1, max_mortor2 = get_main_freq(tsa_fftoutput, ip.mortor, ip.side_band_range)

        def get_sideband_mask(df, main_freq, rng):
            mask1 = df['multiply_freqs'] >= float(main_freq['multiply_freqs']) - rng
            mask2 = df['multiply_freqs'] < float(main_freq['multiply_freqs'])
            mask3 = df['multiply_freqs'] > float(main_freq['multiply_freqs'])
            mask4 = df['multiply_freqs'] <= float(main_freq['multiply_freqs']) + rng
            return (df[mask1 & mask2].index.tolist() +
                    df[mask3 & mask4].index.tolist())

        filter_side_band = []
        filter_side_band += get_sideband_mask(tsa_fftoutput, max_mortor_gear1, ip.mortor_gear_range)
        filter_side_band += get_sideband_mask(tsa_fftoutput, max_mortor_gear2, ip.mortor_gear_range)
        filter_side_band += get_sideband_mask(tsa_fftoutput, max_belt_si1, ip.belt_si_range)
        filter_side_band += get_sideband_mask(tsa_fftoutput, max_belt_si2, ip.belt_si_range)
        filter_side_band += get_sideband_mask(tsa_fftoutput, max_mortor1, ip.morotr_range)
        filter_side_band += get_sideband_mask(tsa_fftoutput, max_mortor2, ip.morotr_range)

        indexlist1 = tsa_fftoutput[tsa_fftoutput['multiply_freqs'].isin(max_harmonic_freq_combine['multiply_freqs'].tolist())].index.tolist()
        indexlist2 = tsa_fftoutput[tsa_fftoutput['multiply_freqs'].isin(max_sideband_freq_combine['multiply_freqs'].tolist())].index.tolist()
        filter_side_band += indexlist1
        filter_side_band += indexlist2

        tsa_fftoutput = tsa_fftoutput.drop(filter_side_band)
        ifft_tsa_output, time_value = fd.ifft_process(tsa_fftoutput['tsa_fft'])
        return tsa_fftoutput, ifft_tsa_output

