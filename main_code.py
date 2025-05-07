import pandas as pd
import datetime
import x_parameter as xo
import y_parameter as yo
import z_parameter as zo
import os

root_inputdir =r'D:\MFPDataset4'

signalList = []
tsaList = []
goodtsaList = []
signalDirList = []
signalDirList_segment = []
numberList = []
numberList_segment = []
chartList = []
chartList_segment = []

#將資料排序且分類的讀取進來
for parents, dirnames, filenames in os.walk(root_inputdir):
     for filename in sorted(filenames):
                
         if (bool('signal' in filename)==True):
           signalList.append(os.path.join(parents,filename))
           dirname = os.path.dirname(os.path.join(parents,filename))
           basename = os.path.basename(dirname) 
           signalDirList.append(basename)
           
           if (bool('C1' in basename)==True):
               chartList.append('C1')
           elif(bool('C2' in basename)==True):
               chartList.append('C2')
           else:
               chartList.append('C4')  
           
           for j in range(1,16):
             if(len(str(j))==1):  
                  if (bool('00' + str(j) in filename)==True):     
                      numberList.append(j)
             else:
                  if (bool('0' + str(j) in filename)==True):     
                      numberList.append(j)
           
           
         if(bool('TSA_60T' in filename) == True):
           tsaList.append(os.path.join(parents,filename))
           
         if(bool('NOL_grease-baseline_M354B_accelerometer_TSA_60T' in filename) == True):
             
            dirname_segment = os.path.dirname(os.path.join(parents,filename))
            basename_segment = os.path.basename(dirname_segment)
            signalDirList_segment.append(basename_segment)
            
            if (bool('C1' in basename_segment)==True):
               chartList_segment.append('C1')
            elif(bool('C2' in basename_segment)==True):
               chartList_segment.append('C2')
            else:
               chartList_segment.append('C4')  
           
            for k in range(1,16):
              if(len(str(k))==1):  
                  if (bool('00' + str(k) in filename)==True):     
                      numberList_segment.append(k)
              else:
                  if (bool('0' + str(k) in filename)==True):     
                      numberList_segment.append(k)
            
            goodtsaList.append(os.path.join(parents,filename))  
            
filenamepdata1 = pd.DataFrame({'signal':signalList})
filenamepdata2 = pd.DataFrame({'tsa':tsaList})
filenamepdata3 = pd.DataFrame({'dirname':signalDirList})
filenamepdata4 = pd.DataFrame({'segment':chartList})
filenamepdata5 = pd.DataFrame({'number':numberList})
filename_combine1 = pd.concat([filenamepdata1,
                              filenamepdata2,
                              filenamepdata3,
                              filenamepdata4,
                              filenamepdata5],
                              axis=1, ignore_index=False) 

filenamepdata_segment_1 = pd.DataFrame({'tsa':goodtsaList})
filenamepdata_segment_2 = pd.DataFrame({'dirname':signalDirList_segment})
filenamepdata_segment_3 = pd.DataFrame({'segment':chartList_segment})
filenamepdata_segment_4 = pd.DataFrame({'number':numberList_segment})
filename_combine2 = pd.concat([filenamepdata_segment_1,
                              filenamepdata_segment_2,
                              filenamepdata_segment_3,
                              filenamepdata_segment_4],
                              axis=1, ignore_index=False)

today = datetime.datetime.now()
year=today.year
month = today.month
day = today.day
hour = today.hour
minute  = today.minute 
datetimeCombine = str(year) + str(month) + str(day) + str(hour) + str(minute)

signalDirList = list(set(signalDirList)) #移除重覆的資料

#將資料分別做計算並且把x,y,z三軸的特徵資料全部組合
for x in signalDirList:
    
    mask1 = filename_combine1['dirname'] == x
    filename_combine1_filter = filename_combine1[mask1]
    all_parameter_combine = pd.DataFrame()  
    x_parameter_combine = pd.DataFrame()
    y_parameter_combine = pd.DataFrame()
    z_parameter_combine = pd.DataFrame()
    for i in range(0,len(filename_combine1_filter)):
        
        inputdir= str(filename_combine1_filter.iloc[i]['signal'])
        tsa_inputdir=str(filename_combine1_filter.iloc[i]['tsa'])
#        dirmane=str(filename_combine1_filter.iloc[i]['dirname'])
    
        mask3 = filename_combine2['segment'] == filename_combine1_filter.iloc[i]['segment']
        mask4 = filename_combine2['number'] == filename_combine1_filter.iloc[i]['number']
        filename_combine2_filter = filename_combine2[mask3 & mask4]
        good_tsa_list = filename_combine2_filter['tsa'].tolist()
        good_tsa_inputdir = ','.join(good_tsa_list)
              
        x_parameter,x_sideband_tsa_fftoutput,x_sideband_ifft_tsa_output=xo.x_output(inputdir,tsa_inputdir,good_tsa_inputdir)
        y_parameter,y_sideband_tsa_fftoutput,y_sideband_ifft_tsa_output=yo.y_output(inputdir,tsa_inputdir,good_tsa_inputdir)
        z_parameter,z_sideband_tsa_fftoutput,z_sideband_ifft_tsa_output=zo.z_output(inputdir,tsa_inputdir,good_tsa_inputdir)
        
        x_parameter_combine = pd.concat([x_parameter_combine,x_parameter],axis=0,sort=False,ignore_index=True)
        y_parameter_combine = pd.concat([y_parameter_combine,y_parameter],axis=0,sort=False,ignore_index=True)
        z_parameter_combine = pd.concat([z_parameter_combine,z_parameter],axis=0,sort=False,ignore_index=True)
        
        all_parameter_combine = pd.concat([x_parameter_combine,y_parameter_combine,z_parameter_combine],axis=1, ignore_index=False) 
        
        
    #    all_sideband_tsa_fftoutput_combine = pd.concat([x_sideband_tsa_fftoutput.iloc[:,1:3],
    #                                                    x_sideband_tsa_fftoutput.iloc[:,4:5],
    #                                                    y_sideband_tsa_fftoutput.iloc[:,4:5],
    #                                                    z_sideband_tsa_fftoutput.iloc[:,4:5]],
    #                                                    axis=1, ignore_index=False)
    #    all_sideband_ifft_tsa_output_combine = pd.concat([x_sideband_ifft_tsa_output.iloc[:,0:2],
    #                                                    y_sideband_ifft_tsa_output.iloc[:,1],
    #                                                    z_sideband_ifft_tsa_output.iloc[:,1]],
    #                                                    axis=1, ignore_index=False)

#    all_parameter_combine.to_csv(datetimeCombine + '_' + str(x) +'_parameter_combine.csv')
    
    #將資料輸出成CSV
    all_parameter_combine.to_csv('.\\excel\\'+str(x) +'_parameter_combine.csv')
    
#all_sideband_tsa_fftoutput_combine.to_csv(datetimeCombine + '_all_sideband_tsa_fftoutput_combine.csv')
#all_sideband_ifft_tsa_output_combine.to_csv(datetimeCombine + '_all_sideband_ifft_tsa_output_combine.csv')
    





