import pandas as pd
import numpy as np
from initialization import InitParameter as ip

ip = ip()
class HarmonicSildband():
    
   def fftoutput(amp,fs):
        fft_value=np.fft.fft(amp) #原始的FFT,是複數
        abs_fft = np.abs(fft_value) #原始的取絕對值的FFT
        abs_fft_n = (np.abs(fft_value/fft_value.size))*2 # 計算用
        abs_fft_segment1 = np.abs(fft_value/fft_value.size)*2 #畫圖用
        if (len(fft_value) % 2 ==0): 
            abs_fft_segment2 = abs_fft_segment1[0:int(fft_value.size/2)]
        else:
            abs_fft_segment2 = abs_fft_segment1[0:int(fft_value.size/2+1)]
        abs_fft_segment2[1:-1] = 2 * abs_fft_segment2[1:-1]

        freqs = np.fft.fftfreq(fft_value.size,1./fs) #all freqs
        fftoutput=pd.DataFrame({'freqs':np.round(freqs,3),
                                'freqs1':np.round(freqs,5),
                                'abs_fft':abs_fft,
                                'abs_fft_n': abs_fft_n,
                                'fft':fft_value})
        return fft_value,abs_fft,freqs,abs_fft_n,fftoutput   
    
   def tsa_fftoutput(amp,fs,fft):
        tsa_fft_value,tsa_abs_fft,tsa_freqs,tsa_abs_fft_n,_ = HarmonicSildband.fftoutput(amp,fs)
        tsa_fftoutput=pd.DataFrame({'tsa_freqs':np.round(tsa_freqs,3),
                                    'tsa_freqs1':np.round(tsa_freqs,5),
                                    'tsa_abs_fft':tsa_abs_fft,
                                    'tsa_abs_fft_n': tsa_abs_fft_n,
                                    'tsa_fft':tsa_fft_value})
    
        fftoutput=fft
        
       
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

        return tsa_fftoutput    
        
    
   def Sildband(tsa_fft):
              
        tsa_fftoutput = tsa_fft

        mask1 = tsa_fftoutput['multiply_freqs']>=ip.mortor-ip.side_band_range
        mask2 = tsa_fftoutput['multiply_freqs']<=ip.mortor+ip.side_band_range
        max_mortor=tsa_fftoutput[mask1 & mask2]
        max_mortor = tsa_fftoutput[tsa_fftoutput['tsa_abs_fft']==np.max(max_mortor['tsa_abs_fft'])]
        max_mortor1=max_mortor.iloc[0:1]
        
#        計算Sideband的頻率，從2.75倍到14.25倍，以0.25逐漸增加
        max_filter_list=[]
        max_filter_freq_list=[]
        max_filter_freq_combine=pd.DataFrame()
        for i in np.arange(2.75,14.25,0.25):
            f1 = tsa_fftoutput['multiply_freqs']>=(np.float(max_mortor1['multiply_freqs']) * i) - ip.high_hamonic_range
            f2 = tsa_fftoutput['multiply_freqs']<(np.float(max_mortor1['multiply_freqs']) * i) + ip.high_hamonic_range
            filter_tsa=tsa_fftoutput[f1 & f2]
            if(filter_tsa.empty==False):
                max_filter=np.max(filter_tsa['tsa_abs_fft_n'])
                max_filter_freq=tsa_fftoutput[tsa_fftoutput['tsa_abs_fft_n']==max_filter].iloc[:,2:5] #取欄位
                max_filter_list.append(max_filter)
                max_filter_freq_list.append(max_filter_freq['multiply_freqs'].tolist())
                #組合每一段倍率的頻段數值
                max_filter_freq_combine = pd.concat([max_filter_freq,max_filter_freq_combine],
                                                    axis=0, ignore_index=False)  
            
        #計算例外的頻率
        f3 = tsa_fftoutput['multiply_freqs']>=(np.float(max_mortor1['multiply_freqs']) * 11.71) - ip.high_hamonic_range
        f4 = tsa_fftoutput['multiply_freqs']<(np.float(max_mortor1['multiply_freqs']) * 11.71) + ip.high_hamonic_range
        filter_tsa_other=tsa_fftoutput[f3 & f4]
        if(filter_tsa_other.empty==False):
            max_filter_other=np.max(filter_tsa_other['tsa_abs_fft_n'])
            max_filter_other_freq=tsa_fftoutput[tsa_fftoutput['tsa_abs_fft_n']==max_filter_other].iloc[:,2:5]
            max_filter_list.append(max_filter_other)
            max_filter_freq_combine = pd.concat([max_filter_other_freq,max_filter_freq_combine],axis=0, ignore_index=False)
        filter_sum = np.sum(max_filter_list)
        
        return filter_sum,max_filter_freq_combine
    
   def Harmonic(fft):
        fftoutput = fft
        
        mask1 = fftoutput['freqs']>=ip.mortor - ip.side_band_range
        mask2 = fftoutput['freqs']<=ip.mortor + ip.side_band_range
        max_mortor=fftoutput[mask1 & mask2]
        max_mortor = fftoutput[fftoutput['abs_fft']==np.max(max_mortor['abs_fft'])]
        max_mortor1=max_mortor.iloc[0:1]
        
        #計算Harmonic的頻率，從0.25倍到2.75倍，以0.25逐漸增加
        max_harmonic_list=[]
        max_harmonic_freq_list=[]
        max_harmonic_freq_combine = pd.DataFrame()
        for i in np.arange(0.25,2.75,0.25):
            f1 = fftoutput['freqs']>=(np.float(max_mortor1['freqs']) * i) - ip.harmonic_gmf_range
            f2 = fftoutput['freqs']<(np.float(max_mortor1['freqs']) * i) + ip.harmonic_gmf_range
            harmonic_filter=fftoutput[f1 & f2]
            if(harmonic_filter.empty==False):
                max_harmonic_filter=np.max(harmonic_filter['abs_fft_n'])
                max_harmonic_freq=fftoutput[fftoutput['abs_fft_n']==max_harmonic_filter].iloc[:,1:3] #取欄位
                max_harmonic_list.append(max_harmonic_filter)
                max_harmonic_freq_list.append(max_harmonic_freq['freqs1'].tolist())
                #組合每一段倍率的頻段數值
                max_harmonic_freq_combine = pd.concat([max_harmonic_freq,max_harmonic_freq_combine],
                                                      axis=0, ignore_index=False)
         
        harmonic_sum = np.sum(max_harmonic_list)    
        return harmonic_sum,max_harmonic_freq_combine
    
   #計算實時同步訊號的Harmonic 
   def Tsa_Harmonic(tsa_fft):
        tsa_fftoutput = tsa_fft
        
        mask1 = tsa_fftoutput['multiply_freqs']>=ip.mortor - ip.side_band_range
        mask2 = tsa_fftoutput['multiply_freqs']<=ip.mortor + ip.side_band_range
        max_mortor=tsa_fftoutput[mask1 & mask2]
        max_mortor = tsa_fftoutput[tsa_fftoutput['tsa_abs_fft']==np.max(max_mortor['tsa_abs_fft'])]
        max_mortor1=max_mortor.iloc[0:1]
        
        max_harmonic_list=[]
        max_harmonic_freq_list=[]
        max_harmonic_freq_combine = pd.DataFrame()
        for i in np.arange(0.25,2.75,0.25):
            f1 = tsa_fftoutput['multiply_freqs']>=(np.float(max_mortor1['multiply_freqs']) * i) - ip.harmonic_gmf_range
            f2 = tsa_fftoutput['multiply_freqs']<(np.float(max_mortor1['multiply_freqs']) * i) + ip.harmonic_gmf_range
            harmonic_filter=tsa_fftoutput[f1 & f2]
            if(harmonic_filter.empty==False):
                max_harmonic_filter=np.max(harmonic_filter['tsa_abs_fft_n'])
                max_harmonic_freq=tsa_fftoutput[tsa_fftoutput['tsa_abs_fft_n']==max_harmonic_filter].iloc[:,2:4]
                max_harmonic_list.append(max_harmonic_filter)
                max_harmonic_freq_list.append(max_harmonic_freq['multiply_freqs'].tolist())
                max_harmonic_freq_combine = pd.concat([max_harmonic_freq,max_harmonic_freq_combine],axis=0, ignore_index=False)
         
        harmonic_sum = np.sum(max_harmonic_list)    
        return harmonic_sum,max_harmonic_freq_combine    
    
    