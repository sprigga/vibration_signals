import pandas as pd
import numpy as np
from initialization import InitParameter as ip
from timedomain import TimeDomain as td
from harmonic_sildband_table import HarmonicSildband as hs

ip=ip()

class FrequencyDomain():

#    計算傅立葉轉換
    def fft_process(amp,fs):
        fft_value=np.fft.fft(amp) #原始的FFT,是複數
        abs_fft = np.abs(fft_value) #原始的取絕對值的FFT
        abs_fft_n = (np.abs(fft_value/fft_value.size))*2 # 計算用
        abs_fft_segment1 = np.abs(fft_value/fft_value.size)*2 #畫圖用
        if (len(fft_value) % 2 ==0): 
            abs_fft_segment2 = abs_fft_segment1[0:int(fft_value.size/2)]
        else:
            abs_fft_segment2 = abs_fft_segment1[0:int(fft_value.size/2+1)]
        abs_fft_segment2[1:-1] = 2 * abs_fft_segment2[1:-1]
        freqs_number_segment = fs*np.arange(0,(fft_value.size/2))/fft_value.size
        freqs = np.fft.fftfreq(fft_value.size,1./fs) #all freqs
        return fft_value,abs_fft,freqs,abs_fft_n,abs_fft_segment2,freqs_number_segment
    
#    計算逆傅立葉轉換
    def ifft_process(fft_value):
        
        # 逆FFT的計算
        ifft_value=np.fft.ifft(fft_value)
        time_value = np.arange(0,np.size(ifft_value)) * (360/ifft_value.size)
        ifft_tsa = pd.DataFrame({'Degree':time_value,'Acc':ifft_value.real})
        return ifft_tsa,time_value

    #計算低頻的FM0數值
    def fft_fm0_si(amp,fs):

        fft_value,abs_fft,freqs,abs_fft_n,_,_ = FrequencyDomain.fft_process(amp,fs)
        fftoutput=pd.DataFrame({'freqs':np.round(freqs,3),
                                'freqs1':np.round(freqs,5),
                                'abs_fft':abs_fft,
                                'abs_fft_n': abs_fft_n,
                                'fft':fft_value})

#        先計算mortor gear的主要頻率
        mask1 = fftoutput['freqs']>=ip.mortor_gear-ip.side_band_range
        mask2 = fftoutput['freqs']<=ip.mortor_gear+ip.side_band_range
        max_mortor_gear=fftoutput[mask1 & mask2]
        max_mortor_gear = fftoutput[fftoutput['abs_fft']==np.max(max_mortor_gear['abs_fft'])]
        max_mortor_gear1=max_mortor_gear.iloc[0:1]

        
#        先計算培林的主要頻率
        mask3 = fftoutput['freqs']>=ip.belt_si - ip.side_band_range
        mask4 = fftoutput['freqs']<=ip.belt_si + ip.side_band_range
        max_belt_si=fftoutput[mask3 & mask4]
        max_belt_si = fftoutput[fftoutput['abs_fft']==np.max(max_belt_si['abs_fft'])]
        max_belt_si1=max_belt_si.iloc[0:1]

          
#        用mortor gear的主要頻率來找出周圍的頻率
        mask7 = fftoutput['freqs']>=np.float(max_mortor_gear1['freqs']) - ip.harmonic_gmf_range
        mask8 = fftoutput['freqs']<np.float(max_mortor_gear1['freqs'])
        mask9 = fftoutput['freqs']>np.float(max_mortor_gear1['freqs'])
        mask10 = fftoutput['freqs']<=np.float(max_mortor_gear1['freqs']) + ip.harmonic_gmf_range
        
#        用培林的主要頻率來找出周圍的頻率
        mask11 = fftoutput['freqs']>=np.float(max_belt_si1['freqs']) - ip.harmonic_gmf_range
        mask12 = fftoutput['freqs']<np.float(max_belt_si1['freqs'])
        mask13 = fftoutput['freqs']>np.float(max_belt_si1['freqs'])
        mask14 = fftoutput['freqs']<=np.float(max_belt_si1['freqs']) + ip.harmonic_gmf_range
        
       #呼叫計算harmonic sildband table的方法
        low_filter_sum,_ = hs.Harmonic(fftoutput)
        
#        篩選motor gear和培林的數值                  
        fft_mgs1=fftoutput[mask7 & mask8]
        fft_mgs2=fftoutput[mask9 & mask10]

        fft_bi1=fftoutput[mask11 & mask12]
        fft_bi2=fftoutput[mask13 & mask14]
        
#        計算低頻的FM0的數值
        low_fm0=td.peak(amp)/low_filter_sum
        
#        計算出motor gear si和belt si的數值        
        total_fft_mgs = (np.sum(fft_mgs1['abs_fft_n']) +  np.sum(fft_mgs2['abs_fft_n']))/(len(fft_mgs1) + len(fft_mgs2))
        total_fft_bi = (np.sum(fft_bi1['abs_fft_n']) + np.sum(fft_bi2['abs_fft_n']))/(len(fft_bi1) + len(fft_bi2))
        
        return fftoutput,total_fft_mgs,total_fft_bi,low_fm0
    
#   計算實時同步訊號(TSA)的高頻FM0
    def tsa_fft_fm0_slf(amp,fs,fft):
         
        tsa_fft_value,tsa_abs_fft,tsa_freqs,tsa_abs_fft_n,_,_ = FrequencyDomain.fft_process(amp,fs)
        tsa_fftoutput=pd.DataFrame({'tsa_freqs':np.round(tsa_freqs,3),
                                    'tsa_freqs1':np.round(tsa_freqs,5),
                                    'tsa_abs_fft':tsa_abs_fft,
                                    'tsa_abs_fft_n': tsa_abs_fft_n,
                                    'tsa_fft':tsa_fft_value})
    
        fftoutput=fft
        
       
#        計算TSA FFT和原始FFT頻率的倍率        
        max1=fftoutput[fftoutput['abs_fft']==np.max(fftoutput['abs_fft'])]
        max2=tsa_fftoutput[tsa_fftoutput['tsa_abs_fft']==np.max(tsa_fftoutput['tsa_abs_fft'])]
        
        max3=max1.iloc[0:1]
        max4=max2.iloc[0:1]  
        
        max_freqs=np.float(max3['freqs1'])/np.float(max4['tsa_freqs1'])
        
        tsa_fftoutput=pd.DataFrame({'tsa_freqs':np.round(tsa_freqs,3),
                                    'tsa_freqs1':np.round(tsa_freqs,5),
                                    'multiply_freqs':np.round(tsa_freqs*max_freqs,5),
                                    'tsa_abs_fft':tsa_abs_fft,
                                    'tsa_abs_fft_n': tsa_abs_fft_n,
                                    'tsa_fft':tsa_fft_value})
        
#        先計算mortor gear的主要頻率
        mask1 = tsa_fftoutput['multiply_freqs']>=ip.mortor_gear-ip.side_band_range
        mask2 = tsa_fftoutput['multiply_freqs']<=ip.mortor_gear+ip.side_band_range
        max_mortor_gear=tsa_fftoutput[mask1 & mask2]
        max_mortor_gear = tsa_fftoutput[tsa_fftoutput['tsa_abs_fft']==np.max(max_mortor_gear['tsa_abs_fft'])]
        max_mortor_gear1=max_mortor_gear.iloc[0:1]
        
#        先計算培林的主要頻率        
        mask3 = tsa_fftoutput['multiply_freqs']>=ip.belt_si - ip.side_band_range
        mask4 = tsa_fftoutput['multiply_freqs']<=ip.belt_si + ip.side_band_range
        max_belt_si=tsa_fftoutput[mask3 & mask4]
        max_belt_si = tsa_fftoutput[tsa_fftoutput['tsa_abs_fft']==np.max(max_belt_si['tsa_abs_fft'])]
        max_belt_si1=max_belt_si.iloc[0:1]
        
#         用mortor gear的主要頻率來找出周圍的頻率
        mask7 = tsa_fftoutput['multiply_freqs']>=np.float(max_mortor_gear1['multiply_freqs']) - ip.mortor_gear_range
        mask8 = tsa_fftoutput['multiply_freqs']<np.float(max_mortor_gear1['multiply_freqs'])
        mask9 = tsa_fftoutput['multiply_freqs']>np.float(max_mortor_gear1['multiply_freqs'])
        mask10 = tsa_fftoutput['multiply_freqs']<=np.float(max_mortor_gear1['multiply_freqs']) + ip.mortor_gear_range
        
#        用培林的主要頻率來找出周圍的頻率
        mask11 = tsa_fftoutput['multiply_freqs']>=np.float(max_belt_si1['multiply_freqs']) - ip.belt_si_range
        mask12 = tsa_fftoutput['multiply_freqs']<np.float(max_belt_si1['multiply_freqs'])
        mask13 = tsa_fftoutput['multiply_freqs']>np.float(max_belt_si1['multiply_freqs'])
        mask14 = tsa_fftoutput['multiply_freqs']<=np.float(max_belt_si1['multiply_freqs']) + ip.belt_si_range
        
         #---high freqency fm0---
        high_filter_sum,_ = hs.Sildband(tsa_fftoutput)
         
         
#        篩選motor gear和培林的數值
        tsa_fft_mgs1=tsa_fftoutput[mask7 & mask8]
        tsa_fft_mgs2=tsa_fftoutput[mask9 & mask10]
#
        tsa_fft_bi1=tsa_fftoutput[mask11 & mask12]
        tsa_fft_bi2=tsa_fftoutput[mask13 & mask14]
         
#        計算高頻的FM0的數值
        high_fm0 = td.peak(amp)/ high_filter_sum
        
#        計算出motor gear si和belt si的數值
        total_tsa_fft_mgs = (np.sum(tsa_fft_mgs1['tsa_abs_fft_n']) +  np.sum(tsa_fft_mgs2['tsa_abs_fft_n']))/td.rms(amp)
        total_tsa_fft_bi = (np.sum(tsa_fft_bi1['tsa_abs_fft_n']) +  np.sum(tsa_fft_bi2['tsa_abs_fft_n']))/td.rms(amp)
    
        return tsa_fftoutput,total_tsa_fft_mgs,total_tsa_fft_bi,high_fm0
    
