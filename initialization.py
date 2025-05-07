import pandas as pd

#設定所有的初始值
class InitParameter():
    
    def __init__(self):
        
        self.tsa_fs=80
        self.fs=50000
        self.tsa_ts = 1./self.tsa_fs #the sampling period, 1/sample rate
        self.ts = 1./self.fs
        
        self.mortor_gear = 99.873
        self.belt_si = 41.6
        self.mortor = 1597.97039
        self.high_hamonic = 20673.84
        
        self.side_band_range=1.664
        self.harmonic_gmf_range= 1.5 #3.5 #2.5  #1.5 #0.5 #0.135
            
        self.mortor_gear_range = 2.5 #1.67 #1.65
        self.belt_si_range = 2.5 #1.67
        self.morotr_range = 8.29
        self.high_hamonic_range = 5
        
        self.indexlist1=[]
        self.indexlist2=[]
        
        self.cwt_scale_max=64
        
        self.stft_hann_nperseg = 128
        self.stft_flattop_nperseg = 256


        #---測試用的資料---
        #    inputdir = r'C:\Users\jasonchien\.spyder-py3\20181121_tsa_good.csv'  #檔案路徑，可修改。 
        #    inputdir2 = r'C:\Users\jasonchien\.spyder-py3\MFP45_600C_NOL_L1_M354B_accelerometer_1.csv'  #good
        inputdir = r'C:\Users\jasonchien\.spyder-py3\20181121_tsa_defect.csv'  #檔案路徑，可修改。 
        inputdir2 = r'C:\Users\jasonchien\.spyder-py3\MFP45_600C_59T_L1_M354B_accelerometer_1.csv'  #defect
        inputdir3 = r'C:\Users\jasonchien\.spyder-py3\Harmonic Sideband Filter for Polo_1.csv'  #檔案路徑，可修改。
        inputdir5 = r'D:\MFPDataset5\20190117_104625_ML_GAP0.1_MFP45_600C_C1_NOISE\MFP45_600C_ML_GAP0.1_M354B_accelerometer_signal_010.csv'
        inputdir6 = r'D:\MFPDataset5\20190117_104625_ML_GAP0.1_MFP45_600C_C1_NOISE\MFP45_600C_ML_GAP0.1_M354B_accelerometer_TSA_60T_010.csv'
        
        dataSet = pd.read_csv(inputdir,names=['Degree','Acc'])
        dataSet2 = pd.read_csv(inputdir2,names=['time','x','y','z','label','12m','60m'])
        dataSet3 = pd.read_csv(inputdir3,header=None)
        self.pdata = pd.DataFrame(dataSet,columns=['Degree','Acc'])
        self.pdata2 = pd.DataFrame(dataSet2,columns=['time','x','y','z','label','12m','60m'])
        self.pdata3 = pd.DataFrame(dataSet3)
        self.pdata4 = self.pdata3[0]*-1
        
        dataSet5 = pd.read_csv(inputdir5,names=['time','x','y','z','label','12m','60m'])
        self.pdata5 = pd.DataFrame(dataSet5,columns=['time','x','y','z','label','12m','60m'])
        
        dataSet6 = pd.read_csv(inputdir6,names=['Degree','x','y','z'])
        self.pdata6 = pd.DataFrame(dataSet6,columns=['Degree','x','y','z'])
        #----------------
