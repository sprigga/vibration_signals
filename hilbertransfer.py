import pandas as pd
import numpy as np
from scipy.signal import hilbert, chirp

class HilberTransfer():
    
    #計算NB4數值
    def NB4(amp,m):
            
            lastDegree=amp['Degree'].iloc[-1] #先取最後一筆角度的數值
            segment=m #資料要分割成多少段
            degreeSegment=np.arange(0,lastDegree+(lastDegree/segment),(lastDegree/segment)) #每個資料的區段範圍
            ampMean=amp['hilbert_envelope'].agg(np.mean) #計算振幅的平均值
            allAmpCount=len(amp) #資料的總長度
            totalSumSegment=0
            
            #先計算NB4分子的部分
            for i in range(1,len(degreeSegment)):
    
               mask1=amp['Degree']>degreeSegment[i-1]
               mask2=amp['Degree']<=degreeSegment[i]
               amp2 = amp[mask1 & mask2]
               amp2Mean=amp2['hilbert_envelope'].agg(np.mean)
               amp3=pd.DataFrame({'Degree':amp2['Degree'],'hilbert_envelope':amp2['hilbert_envelope'],'Mean':amp2Mean})
               sumSegment=sum((np.float64(amp3['hilbert_envelope'])-np.float64(amp3['Mean']))**2) / len(amp3['hilbert_envelope']) #分母
               totalSumSegment = sumSegment + totalSumSegment
            
            #計算NB4分母的部分
            divisionTotalSumSegment=(totalSumSegment/segment)**2
            totalSumAll = sum((np.float64(amp['hilbert_envelope'])-ampMean)**4) / allAmpCount #分子
            nb4 = totalSumAll/divisionTotalSumSegment
            
            return nb4
     
    #計算hibler Transfer數值    
    def ht(amp,label):
        
        analytic_signal = hilbert(amp[label]) #要分析某個軸的訊號
        amplitude_envelope = np.abs(analytic_signal)
        htpdata = pd.DataFrame({'Degree':amp['Degree'],'hilbert_real':analytic_signal.real,
                               'hilbert_imag':analytic_signal.imag,'hilbert_envelope':amplitude_envelope})
        return htpdata
    

  