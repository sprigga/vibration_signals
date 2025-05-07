import pandas as pd
import numpy as np
from timedomain import TimeDomain as td
from frequencydomain import FrequencyDomain as fd
from filterprocess import FilterProcess as fp
from waveletprocess import WaveLetProcess as wp
from hilbertransfer import HilberTransfer as ht
from initialization import InitParameter as ip
import pywt
#import os
ip=ip()

def x_output(inputdir,tsa_inputdir,good_tsa_inputdir):
    
    #---計算x軸23個特徵值---
    x_peakList = []
    x_avgList = []
    x_rmsList = []
    x_kurtList = []
    x_cfList = []
    
    x_tsaPeakList = []
    x_tsaRmsList = []
    x_tsaEoList = []
    
    x_fftFm0List = []
    x_fftMotorGearList = []
    x_fftBeltList = []
    
    x_tsa_fftFm0List = []
    x_tsa_fftMotorGearList = []
    x_tsa_fftBeltList = []
    
    x_harmonic_na4=[]
    x_harmonic_na4s=[]
    
    x_sideband_fm4=[]
    x_sideband_m6A=[]
    x_sideband_m8a=[]
    x_sideband_er=[]
    
    x_ht_nb4=[]
    x_flat_np4=[]
    x_cwt_np4=[]
    
    good_tsa_dataSet = pd.read_csv(good_tsa_inputdir,names=['Degree','x','y','z'])
    good_tsa_pdata = pd.DataFrame(good_tsa_dataSet,columns=['Degree','x','y','z'])
          
    dataSet = pd.read_csv(inputdir,names=['time','x','y','z','label','12m','60m'])
    pdata = pd.DataFrame(dataSet,columns=['time','x','y','z','label','12m','60m'])
    
    x_peak = td.peak(pdata['x'])
    x_avg = td.avg(pdata['x'])
    x_rms = td.rms(pdata['x'])
    x_kurt = td.kurt(pdata['x'])
    x_cf = td.cf(pdata['x'])
    
    fftoutput,total_fft_mgs,total_fft_bi,low_fm0 = fd.fft_fm0_si(pdata['x'],ip.fs)
    
    flat_np4,hann_np4  = wp.StftProcess(pdata['x'],ip.fs)
    
    coef,freqs=pywt.cwt(pdata['x'],np.arange(1,ip.cwt_scale_max),'db8',ip.ts)
    cwt_np4 = wp.CWTProcess(coef,freqs)
    
    x_peakList.append(x_peak)
    x_avgList.append(x_avg)
    x_rmsList.append(x_rms)
    x_kurtList.append(x_kurt)
    x_cfList.append(x_cf)
    
    x_fftFm0List.append(low_fm0)
    x_fftMotorGearList.append(total_fft_mgs)
    x_fftBeltList.append(total_fft_bi)
    
    x_flat_np4.append(flat_np4)
    x_cwt_np4.append(cwt_np4)
    
    
    tsa_dataSet = pd.read_csv(tsa_inputdir,names=['Degree','x','y','z'])
    tsa_pdata = pd.DataFrame(tsa_dataSet)
     
    x_tsa_peak = td.peak(tsa_pdata['x'])
    x_tsa_rms = td.rms(tsa_pdata['x'])
    x_tsa_eo = td.eo(tsa_pdata,'x')
    
    tsa_fftoutput,total_tsa_fft_mgs,total_tsa_fft_bi,high_fm0 = fd.tsa_fft_fm0_slf(tsa_pdata['x'],ip.tsa_fs,fftoutput)
    
    ifft_tsa_output = fp.FilterHarmonic(tsa_pdata['x'],ip.tsa_fs,fftoutput)
    good_ifft_tsa = fp.FilterHarmonic(good_tsa_pdata['x'],ip.tsa_fs,fftoutput)
    har_na4,_,_ = fp.NA4(ifft_tsa_output,2)
    har_na4s,_ = fp.NA4S(ifft_tsa_output,good_ifft_tsa,2)
    
    sideband_tsa_fftoutput,sideband_ifft_tsa_output = fp.FilterSideband(tsa_pdata['x'],ip.tsa_fs,fftoutput)
    fm4 = fp.FM4(sideband_ifft_tsa_output['Acc'])
    m6a = fp.M6A(sideband_ifft_tsa_output['Acc'])
    m8a = fp.M8A(sideband_ifft_tsa_output['Acc'])
    er = fp.ER(tsa_pdata['x'],ip.tsa_fs,fftoutput)
    #        
    htpdata = ht.ht(tsa_pdata,'x')
    nb4 = ht.NB4(htpdata,2)
    #        
    x_tsaPeakList.append(x_tsa_peak)
    x_tsaRmsList.append(x_tsa_rms)
    x_tsaEoList.append(x_tsa_eo)
    
    x_tsa_fftFm0List.append(high_fm0)
    x_tsa_fftMotorGearList.append(total_tsa_fft_mgs)
    x_tsa_fftBeltList.append(total_tsa_fft_bi)
    
    x_harmonic_na4.append(har_na4)
    x_harmonic_na4s.append(har_na4s)
    
    x_sideband_fm4.append(fm4)
    x_sideband_m6A.append(m6a)
    x_sideband_m8a.append(m8a)
    x_sideband_er.append(er)
    
    x_ht_nb4.append(nb4)
    
    #把x軸特徵值的資料全部組合
    x_parameter  = pd.DataFrame({'x_peak':x_peakList,'x_avg':x_avgList,'x_rms':x_rmsList,'x_kurt':x_kurtList,'x_cf':x_cfList,
                                   'x_tsaPeakList':x_tsaPeakList,'x_tsaRmsList':x_tsaRmsList,'x_tsaEoList':x_tsaEoList,
                                   'x_fftFm0List':x_fftFm0List,'x_fftMotorGearList':x_fftMotorGearList,'x_fftBeltList':x_fftBeltList,
                                   'x_tsa_fftFm0List':x_tsa_fftFm0List,'x_tsa_fftMotorGearList':x_tsa_fftMotorGearList,'x_tsa_fftBeltList':x_tsa_fftBeltList,
                                   'x_harmonic_na4':x_harmonic_na4,'x_harmonic_na4s':x_harmonic_na4s,
                                   'x_sideband_fm4':x_sideband_fm4,'x_sideband_m6A':x_sideband_m6A,'x_sideband_m8a':x_sideband_m8a,'x_sideband_er':x_sideband_er,
                                   'x_ht_nb4':x_ht_nb4,'x_flat_np4':x_flat_np4,'x_cwt_np4':x_cwt_np4})
    
    return x_parameter,sideband_tsa_fftoutput,sideband_ifft_tsa_output
