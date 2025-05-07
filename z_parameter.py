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

def z_output(inputdir, tsa_inputdir, good_tsa_inputdir):
    z_peakList = []
    z_avgList = []
    z_rmsList = []
    z_kurtList = []
    z_cfList = []

    z_tsaPeakList = []
    z_tsaRmsList = []
    z_tsaEoList = []

    z_fftFm0List = []
    z_fftMotorGearList = []
    z_fftBeltList = []

    z_tsa_fftFm0List = []
    z_tsa_fftMotorGearList = []
    z_tsa_fftBeltList = []

    z_harmonic_na4 = []
    z_harmonic_na4s = []

    z_sideband_fm4 = []
    z_sideband_m6A = []
    z_sideband_m8a = []
    z_sideband_er = []

    z_ht_nb4 = []
    z_flat_np4 = []
    z_cwt_np4 = []

    filenameList = []

    labelNameList1 = []
    labelNumList1 = []
    labelNameList2 = []
    labelNumList2 = []

    good_tsa_dataSet = pd.read_csv(good_tsa_inputdir, names=['Degree', 'x', 'y', 'z'])
    good_tsa_pdata = pd.DataFrame(good_tsa_dataSet, columns=['Degree', 'x', 'y', 'z'])

    dataSet = pd.read_csv(inputdir, names=['time', 'x', 'y', 'z', 'label', '12m', '60m'])
    pdata = pd.DataFrame(dataSet, columns=['time', 'x', 'y', 'z', 'label', '12m', '60m'])

    z_peak = td.peak(pdata['z'])
    z_avg = td.avg(pdata['z'])
    z_rms = td.rms(pdata['z'])
    z_kurt = td.kurt(pdata['z'])
    z_cf = td.cf(pdata['z'])

    fftoutput, total_fft_mgs, total_fft_bi, low_fm0 = fd.fft_fm0_si(pdata['z'], ip.fs)

    flat_np4, hann_np4 = wp.StftProcess(pdata['z'], ip.fs)

    coef, freqs = pywt.cwt(pdata['z'], np.arange(1, ip.cwt_scale_max), 'db8', ip.ts)
    cwt_np4 = wp.CWTProcess(coef, freqs)

    z_peakList.append(z_peak)
    z_avgList.append(z_avg)
    z_rmsList.append(z_rms)
    z_kurtList.append(z_kurt)
    z_cfList.append(z_cf)

    z_fftFm0List.append(low_fm0)
    z_fftMotorGearList.append(total_fft_mgs)
    z_fftBeltList.append(total_fft_bi)

    z_flat_np4.append(flat_np4)
    z_cwt_np4.append(cwt_np4)

    tsa_dataSet = pd.read_csv(tsa_inputdir, names=['Degree', 'x', 'y', 'z'])
    tsa_pdata = pd.DataFrame(tsa_dataSet)

    z_tsa_peak = td.peak(tsa_pdata['z'])
    z_tsa_rms = td.rms(tsa_pdata['z'])
    z_tsa_eo = td.eo(tsa_pdata, 'z')

    tsa_fftoutput, total_tsa_fft_mgs, total_tsa_fft_bi, high_fm0 = fd.tsa_fft_fm0_slf(tsa_pdata['z'], ip.tsa_fs, fftoutput)

    ifft_tsa_output = fp.FilterHarmonic(tsa_pdata['z'], ip.tsa_fs, fftoutput)
    good_ifft_tsa = fp.FilterHarmonic(good_tsa_pdata['z'], ip.tsa_fs, fftoutput)
    har_na4, _, _ = fp.NA4(ifft_tsa_output, 2)
    har_na4s, _ = fp.NA4S(ifft_tsa_output, good_ifft_tsa, 2)

    sideband_tsa_fftoutput, sideband_ifft_tsa_output = fp.FilterSideband(tsa_pdata['z'], ip.tsa_fs, fftoutput)
    fm4 = fp.FM4(sideband_ifft_tsa_output['Acc'])
    m6a = fp.M6A(sideband_ifft_tsa_output['Acc'])
    m8a = fp.M8A(sideband_ifft_tsa_output['Acc'])
    er = fp.ER(tsa_pdata['z'], ip.tsa_fs, fftoutput)

    htpdata = ht.ht(tsa_pdata, 'z')
    nb4 = ht.NB4(htpdata, 2)

    z_tsaPeakList.append(z_tsa_peak)
    z_tsaRmsList.append(z_tsa_rms)
    z_tsaEoList.append(z_tsa_eo)

    z_tsa_fftFm0List.append(high_fm0)
    z_tsa_fftMotorGearList.append(total_tsa_fft_mgs)
    z_tsa_fftBeltList.append(total_tsa_fft_bi)

    z_harmonic_na4.append(har_na4)
    z_harmonic_na4s.append(har_na4s)

    z_sideband_fm4.append(fm4)
    z_sideband_m6A.append(m6a)
    z_sideband_m8a.append(m8a)
    z_sideband_er.append(er)

    z_ht_nb4.append(nb4)

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

    z_parameter = pd.DataFrame({
        'z_peak': z_peakList, 'z_avg': z_avgList, 'z_rms': z_rmsList, 'z_kurt': z_kurtList, 'z_cf': z_cfList,
        'z_tsaPeakList': z_tsaPeakList, 'z_tsaRmsList': z_tsaRmsList, 'z_tsaEoList': z_tsaEoList,
        'z_fftFm0List': z_fftFm0List, 'z_fftMotorGearList': z_fftMotorGearList, 'z_fftBeltList': z_fftBeltList,
        'z_tsa_fftFm0List': z_tsa_fftFm0List, 'z_tsa_fftMotorGearList': z_tsa_fftMotorGearList, 'z_tsa_fftBeltList': z_tsa_fftBeltList,
        'z_harmonic_na4': z_harmonic_na4, 'z_harmonic_na4s': z_harmonic_na4s,
        'z_sideband_fm4': z_sideband_fm4, 'z_sideband_m6A': z_sideband_m6A, 'z_sideband_m8a': z_sideband_m8a, 'z_sideband_er': z_sideband_er,
        'z_ht_nb4': z_ht_nb4, 'z_flat_np4': z_flat_np4, 'z_cwt_np4': z_cwt_np4, 'filename': filenameList,
        'label_Name_1': labelNameList1, 'label_Num_1': labelNumList1, 'label_Name_2': labelNameList2, 'label_Num_2': labelNumList2
    })

    return z_parameter, sideband_tsa_fftoutput, sideband_ifft_tsa_output

