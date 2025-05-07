import pandas as pd
import numpy as np
from math import sqrt
from scipy.stats import kurtosis

class TimeDomain:
    @staticmethod
    def peak(x):
        return np.max(x) - np.min(x)

    @staticmethod
    def avg(x):
        return np.average(x)

    @staticmethod
    def rms(num):
        return sqrt(np.mean(np.square(num)))

    @staticmethod
    def cf(num):
        rms = np.sqrt(np.mean(np.square(num)))
        return (np.max(num) - np.min(num)) / rms if rms != 0 else np.nan

    @staticmethod
    def kurt(num):
        # 使用 scipy.stats.kurtosis，fisher=False 保持與原公式一致
        return kurtosis(num, fisher=False, bias=False)

    @staticmethod
    def eo(num, label):
        # 用 pd.concat 取代 append
        num2 = pd.concat([num, num.iloc[[0]]], ignore_index=True)
        pdatadelta1 = (num2[label].shift() ** 2) - (num2[label] ** 2)
        pdatadelta1 = pdatadelta1.dropna()
        pdatadelta2 = num[label].agg(np.sum) / len(num)
        pdata_combine = pd.DataFrame({'delta1': pdatadelta1, 'delta2': pdatadelta2})
        pdatadelta3 = pdata_combine['delta1'] - pdata_combine['delta2']
        return ((len(pdatadelta3) ** 2) * np.sum(pdatadelta3 ** 4)) / (np.sum(pdatadelta3 ** 2) ** 2)

