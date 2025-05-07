import pandas as pd
import numpy as np
from initialization import InitParameter as ip
from scipy import signal
from typing import Tuple, Any

ip = ip()

# 頻率區間常數
FREQ_LOW = 800
FREQ_HIGH = 2500

class WaveLetProcess:
    @staticmethod
    def NP4(z: np.ndarray) -> float:
        Zxx_mean = np.mean(z)
        Zxx_diff = z - Zxx_mean
        Zxx_power2 = np.power(Zxx_diff, 2)
        Zxx_sum2 = np.sum(Zxx_power2)
        Zxx_square = np.power(Zxx_sum2, 2)
        Zxx_power4 = np.power(Zxx_diff, 4)
        Zxx_sum4 = np.sum(Zxx_power4)
        Zxx_multiply = z.size * Zxx_sum4
        np4 = Zxx_multiply / Zxx_square if Zxx_square != 0 else np.nan
        return np4

    @staticmethod
    def StftProcess(amp: np.ndarray, fs: int) -> Tuple[float, float]:
        try:
            hann_nperseg = ip.stft_hann_nperseg
            flattop_nperseg = ip.stft_flattop_nperseg

            # Hann window
            hann_win_128 = signal.get_window('hann', hann_nperseg, fftbins=False)
            f_hann, t_hann, z_hann = signal.stft(amp, fs, hann_win_128, hann_nperseg, noverlap=int(hann_nperseg * 0.95))
            abs_z_hann = np.abs(z_hann)
            z_hann_pdata = pd.DataFrame(abs_z_hann)
            f_hann_pdata = pd.DataFrame({'f': f_hann})
            hann_filter = (f_hann_pdata['f'] > FREQ_LOW) & (f_hann_pdata['f'] <= FREQ_HIGH)
            f_hann_pdata_filter = f_hann_pdata[hann_filter].index.tolist()
            z_hann_pdata_filter = z_hann_pdata.iloc[f_hann_pdata_filter]
            hann_np4 = WaveLetProcess.NP4(np.float64(z_hann_pdata_filter))

            # Flattop window
            flattop_win_256 = signal.get_window('flattop', flattop_nperseg, fftbins=False)
            f_flat, t_flat, z_flat = signal.stft(amp, fs, flattop_win_256, flattop_nperseg, noverlap=int(flattop_nperseg * 0.95))
            abs_z_flat = np.abs(z_flat)
            z_flat_pdata = pd.DataFrame(abs_z_flat)
            f_flat_pdata = pd.DataFrame({'f': f_flat})
            flat_filter = (f_flat_pdata['f'] > FREQ_LOW) & (f_flat_pdata['f'] <= FREQ_HIGH)
            f_flat_pdata_filter = f_flat_pdata[flat_filter].index.tolist()
            z_flat_pdata_filter = z_flat_pdata.iloc[f_flat_pdata_filter]
            flat_np4 = WaveLetProcess.NP4(np.float64(z_flat_pdata_filter))

            return flat_np4, hann_np4
        except Exception as e:
            print(f"StftProcess error: {e}")
            return np.nan, np.nan

    @staticmethod
    def CWTProcess(coef: np.ndarray, freqs: np.ndarray) -> float:
        try:
            coef_pdata = pd.DataFrame(coef)
            freqs_pdata = pd.DataFrame({'f': freqs})
            freqs_filter = (freqs_pdata['f'] > FREQ_LOW) & (freqs_pdata['f'] <= FREQ_HIGH)
            freqs_pdata_filter = freqs_pdata[freqs_filter].index.tolist()
            coef_pdata_filter = coef_pdata.iloc[freqs_pdata_filter]
            cwt_np4 = WaveLetProcess.NP4(np.float64(coef_pdata_filter))
            return cwt_np4
        except Exception as e:
            print(f"CWTProcess error: {e}")
            return np.nan

    @staticmethod
    def NE(coef: np.ndarray, freqs: np.ndarray, t: np.ndarray) -> Any:
        try:
            freqs_coef_segment_sum_list = []
            time_coef_segment_sum_list = []

            coef_pdata = pd.DataFrame(np.abs(coef))
            freqs_pdata = pd.DataFrame({'f': freqs})
            time_pdata = pd.DataFrame({'t': t})

            #---計算要切割的時間區段和頻率區段---
            time_pdata_filter = time_pdata[time_pdata['t'] <= 0.5]
            time_pdata_filter_value = time_pdata[time_pdata['t'] == 0.5].index.values
            time_pdata_filter_list = time_pdata[time_pdata['t'] <= 0.5].index.values
            time_pdata_division = np.round(len(time_pdata) / time_pdata_filter_value) if len(time_pdata_filter_value) > 0 else 1
            time_pdata_division_list = pd.DataFrame({'t1': time_pdata_division - (time_pdata.size / time_pdata_filter_list)}) if len(time_pdata_filter_list) > 0 else pd.DataFrame({'t1': []})
            time_pdata_subtract = pd.concat([time_pdata_filter, time_pdata_division_list], axis=1, ignore_index=False)
            time_pdata_subtract_filter = time_pdata_subtract[time_pdata_subtract['t1'].between(0, 0.00005)].index.values if 't1' in time_pdata_subtract else []
            timeSegment = np.arange(0, len(time_pdata) + len(time_pdata_subtract_filter), len(time_pdata_subtract_filter)) if len(time_pdata_subtract_filter) > 0 else np.array([0, len(time_pdata)])
            freqSegment = np.arange(FREQ_HIGH, 500, -200)
            #-------------------------------------
            #分別計算分割時間區段的每個數值，和分割頻率區段的每個數值，二者相除
            for i in range(1, len(freqSegment)):
                freq_segment_division_list = []
                freqs_filter = (freqs_pdata['f'] > freqSegment[i]) & (freqs_pdata['f'] <= freqSegment[i - 1])
                freqs_pdata_filter = freqs_pdata[freqs_filter].index.tolist()
                freqs_coef_pdata_filter = coef_pdata.iloc[freqs_pdata_filter]
                freqs_coef_segment_sum = np.sum(np.float64(freqs_coef_pdata_filter))
                freqs_coef_segment_sum_list.append(freqs_coef_segment_sum)
                for j in range(1, len(timeSegment)):
                    time_filter1 = time_pdata.index >= timeSegment[j - 1]
                    time_filter2 = time_pdata.index < timeSegment[j]
                    time_pdata_filter = time_pdata[time_filter1 & time_filter2].index.tolist()
                    time_coef_pdata_filter = coef_pdata.loc[freqs_pdata_filter, time_pdata_filter]
                    time_coef_segment_sum = np.sum(np.float64(time_coef_pdata_filter))
                    time_coef_segment_sum_list.append(time_coef_segment_sum)
                    freq_segment_division_list.append(time_coef_segment_sum / freqs_coef_segment_sum if freqs_coef_segment_sum != 0 else np.nan)
                total_segment_sum_pdata = pd.DataFrame({'ne': freq_segment_division_list})

            return total_segment_sum_pdata
        except Exception as e:
            print(f"NE error: {e}")
            return None

