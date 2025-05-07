import pandas as pd
import numpy as np
from timedomain import TimeDomain as td
from frequencydomain import FrequencyDomain as fd
from filterprocess import FilterProcess as fp
from waveletprocess import WaveLetProcess as wp
from hilbertransfer import HilberTransfer as ht
from initialization import InitParameter as ip
import pywt

ip = ip()

def y_output(inputdir, tsa_inputdir, good_tsa_inputdir):
    y_peakList = []
    y_avgList = []
    y_rmsList = []
    y_kurtList = []
    y_cfList = []

    y_tsaPeakList = []
    y_tsaRmsList = []
    y_tsaEoList = []

    y_fftFm0List = []
    y_fftMotorGearList = []
    y_fftBeltList = []

    y_tsa_fftFm0List = []
    y_tsa_fftMotorGearList = []
    y_tsa_fftBeltList = []

    y_harmonic_na4 = []
    y_harmonic_na4s = []

    y_sideband_fm4 = []
    y_sideband_m6A = []
    y_sideband_m8a = []
    y_sideband_er = []

    y_ht_nb4 = []
    y_flat_np4 = []
    y_cwt_np4 = []

    filenameList = []

    labelNameList1 = []
    labelNumList1 = []
    labelNameList2 = []
    labelNumList2 = []

    good_tsa_dataSet = pd.read_csv(good_tsa_inputdir, names=['Degree', 'x', 'y', 'z'])
    good_tsa_pdata = pd.DataFrame(good_tsa_dataSet, columns=['Degree', 'x', 'y', 'z'])

    dataSet = pd.read_csv(inputdir, names=['time', 'x', 'y', 'z', 'label', '12m', '60m'])
    pdata = pd.DataFrame(dataSet, columns=['time', 'x', 'y', 'z', 'label', '12m', '60m'])

    y_peak = td.peak(pdata['y'])
    y_avg = td.avg(pdata['y'])
    y_rms = td.rms(pdata['y'])
    y_kurt = td.kurt(pdata['y'])
    y_cf = td.cf(pdata['y'])

    fftoutput, total_fft_mgs, total_fft_bi, low_fm0 = fd.fft_fm0_si(pdata['y'], ip.fs)

    flat_np4, hann_np4 = wp.StftProcess(pdata['y'], ip.fs)

    coef, freqs = pywt.cwt(pdata['y'], np.arange(1, ip.cwt_scale_max), 'db8', ip.ts)
    cwt_np4 = wp.CWTProcess(coef, freqs)

    y_peakList.append(y_peak)
    y_avgList.append(y_avg)
    y_rmsList.append(y_rms)
    y_kurtList.append(y_kurt)
    y_cfList.append(y_cf)

    y_fftFm0List.append(low_fm0)
    y_fftMotorGearList.append(total_fft_mgs)
    y_fftBeltList.append(total_fft_bi)

    y_flat_np4.append(flat_np4)
    y_cwt_np4.append(cwt_np4)

    tsa_dataSet = pd.read_csv(tsa_inputdir, names=['Degree', 'x', 'y', 'z'])
    tsa_pdata = pd.DataFrame(tsa_dataSet)

    y_tsa_peak = td.peak(tsa_pdata['y'])
    y_tsa_rms = td.rms(tsa_pdata['y'])
    y_tsa_eo = td.eo(tsa_pdata, 'y')

    tsa_fftoutput, total_tsa_fft_mgs, total_tsa_fft_bi, high_fm0 = fd.tsa_fft_fm0_slf(tsa_pdata['y'], ip.tsa_fs, fftoutput)

    ifft_tsa_output = fp.FilterHarmonic(tsa_pdata['y'], ip.tsa_fs, fftoutput)
    good_ifft_tsa = fp.FilterHarmonic(good_tsa_pdata['y'], ip.tsa_fs, fftoutput)
    har_na4, _, _ = fp.NA4(ifft_tsa_output, 2)
    har_na4s, _ = fp.NA4S(ifft_tsa_output, good_ifft_tsa, 2)

    sideband_tsa_fftoutput, sideband_ifft_tsa_output = fp.FilterSideband(tsa_pdata['y'], ip.tsa_fs, fftoutput)
    fm4 = fp.FM4(sideband_ifft_tsa_output['Acc'])
    m6a = fp.M6A(sideband_ifft_tsa_output['Acc'])
    m8a = fp.M8A(sideband_ifft_tsa_output['Acc'])
    er = fp.ER(tsa_pdata['y'], ip.tsa_fs, fftoutput)

    htpdata = ht.ht(tsa_pdata, 'y')
    nb4 = ht.NB4(htpdata, 2)

    y_tsaPeakList.append(y_tsa_peak)
    y_tsaRmsList.append(y_tsa_rms)
    y_tsaEoList.append(y_tsa_eo)

    y_tsa_fftFm0List.append(high_fm0)
    y_tsa_fftMotorGearList.append(total_tsa_fft_mgs)
    y_tsa_fftBeltList.append(total_tsa_fft_bi)

    y_harmonic_na4.append(har_na4)
    y_harmonic_na4s.append(har_na4s)

    y_sideband_fm4.append(fm4)
    y_sideband_m6A.append(m6a)
    y_sideband_m8a.append(m8a)
    y_sideband_er.append(er)

    y_ht_nb4.append(nb4)

    filenameList.append(inputdir)

    # 將label做分類
    if 'LD' in inputdir:
        labelNameList1.append('LD')
        labelNumList1.append('1')
        if 'GOOD' in inputdir:
            labelNameList2.append('GOOD')
            labelNumList2.append('1')
        elif 'NOISE' in inputdir:
            labelNameList2.append('NOISE')
            labelNumList2.append('2')
        elif 'NG' in inputdir:
            labelNameList2.append('NG')
            labelNumList2.append('2')

    if 'NOL' in inputdir:
        labelNameList1.append('NOL')
        labelNumList1.append('2')
        if 'GOOD' in inputdir:
            labelNameList2.append('GOOD')
            labelNumList2.append('1')

    if 'BTF' in inputdir:
        labelNameList1.append('BTF_PLUS')
        labelNumList1.append('3')
        if 'GOOD' in inputdir:
            labelNameList2.append('GOOD')
            labelNumList2.append('1')
        elif 'NOISE' in inputdir:
            labelNameList2.append('NOISE')
            labelNumList2.append('3')
        elif 'NG' in inputdir:
            labelNameList2.append('NG')
            labelNumList2.append('3')

    if 'ML_GAP' in inputdir:
        labelNameList1.append('ML_GAP')
        labelNumList1.append('4')
        if 'GOOD' in inputdir:
            labelNameList2.append('GOOD')
            labelNumList2.append('1')
        elif 'NOISE' in inputdir:
            labelNameList2.append('NOISE')
            labelNumList2.append('4')
        elif 'NG' in inputdir:
            labelNameList2.append('NG')
            labelNumList2.append('4')

    if 'BF' in inputdir:
        labelNameList1.append('BF_R')
        labelNumList1.append('5')
        if 'GOOD' in inputdir:
            labelNameList2.append('GOOD')
            labelNumList2.append('1')
        elif 'NOISE' in inputdir:
            labelNameList2.append('NOISE')
            labelNumList2.append('5')
        elif 'NG' in inputdir:
            labelNameList2.append('NG')
            labelNumList2.append('5')

    if 'BS_MINUS' in inputdir:
        labelNameList1.append('BS_MINUS')
        labelNumList1.append('6')
        if 'GOOD' in inputdir:
            labelNameList2.append('GOOD')
            labelNumList2.append('1')
        elif 'NOISE' in inputdir:
            labelNameList2.append('NOISE')
            labelNumList2.append('6')
        elif 'NG' in inputdir:
            labelNameList2.append('NG')
            labelNumList2.append('6')

    if 'GB' in inputdir:
        labelNameList1.append('GB')
        labelNumList1.append('7')
        if 'GOOD' in inputdir:
            labelNameList2.append('GOOD')
            labelNumList2.append('1')
        elif 'NOISE' in inputdir:
            labelNameList2.append('NOISE')
            labelNumList2.append('7')
        elif 'NG' in inputdir:
            labelNameList2.append('NG')
            labelNumList2.append('7')

    y_parameter = pd.DataFrame({
        'y_peak': y_peakList, 'y_avg': y_avgList, 'y_rms': y_rmsList, 'y_kurt': y_kurtList, 'y_cf': y_cfList,
        'y_tsaPeakList': y_tsaPeakList, 'y_tsaRmsList': y_tsaRmsList, 'y_tsaEoList': y_tsaEoList,
        'y_fftFm0List': y_fftFm0List, 'y_fftMotorGearList': y_fftMotorGearList, 'y_fftBeltList': y_fftBeltList,
        'y_tsa_fftFm0List': y_tsa_fftFm0List, 'y_tsa_fftMotorGearList': y_tsa_fftMotorGearList, 'y_tsa_fftBeltList': y_tsa_fftBeltList,
        'y_harmonic_na4': y_harmonic_na4, 'y_harmonic_na4s': y_harmonic_na4s,
        'y_sideband_fm4': y_sideband_fm4, 'y_sideband_m6A': y_sideband_m6A, 'y_sideband_m8a': y_sideband_m8a, 'y_sideband_er': y_sideband_er,
        'y_ht_nb4': y_ht_nb4, 'y_flat_np4': y_flat_np4, 'y_cwt_np4': y_cwt_np4, 'filename': filenameList,
        'label_Name_1': labelNameList1, 'label_Num_1': labelNumList1, 'label_Name_2': labelNameList2, 'label_Num_2': labelNumList2
    })

    return y_parameter, sideband_tsa_fftoutput, sideband_ifft_tsa_output

