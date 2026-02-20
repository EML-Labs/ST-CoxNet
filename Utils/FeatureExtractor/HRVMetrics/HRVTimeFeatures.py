import sys
import os
from time import time
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

import neurokit2 as nk
import numpy as np
from Utils.FeatureExtractor.Base import BaseExtractor
from neurokit2.hrv.hrv_utils import _hrv_format_input

class HRVTimeFeatures(BaseExtractor):
    def __init__(self, data, fs, rri_given=False):
        self.data = data
        self.fs = fs
        if rri_given:
            self.rri = data # Assuming data is already in miliseconds.
            self.peaks = nk.intervals_to_peaks(self.rri, sampling_rate=self.fs)
        else:
            self.peaks, _ = nk.ecg_peaks(data, sampling_rate=fs)
            self.rri, _, _ = _hrv_format_input(self.peaks, sampling_rate=fs)
    
    def NN50(self):
        diff_rri = np.diff(self.rri)
        nn50_count = np.sum(np.abs(diff_rri) > 50)
        return nn50_count
    
    def pNN31(self):
        diff_rri = np.diff(self.rri)
        nn31_count = np.sum(np.abs(diff_rri) > 31)
        return nn31_count / (len(diff_rri) + 1) * 100
    
    def HR(self):
        hr = 60000 / self.rri
        return {
            "Mean_HR": np.mean(hr), 
            "SD_HR": np.std(hr, ddof=1), 
            "Min_HR": np.min(hr),
            "Max_HR": np.max(hr)
        }
    
    def compute(self):
        hrv_time = nk.hrv_time(self.peaks, sampling_rate=self.fs, show=False)
        features = hrv_time.to_dict(orient='records')[0] if not hrv_time.empty else {}
        features["NN50"] = self.NN50()
        features["pNN31"] = self.pNN31()
        features.update(self.HR())
        return features
    

    def __helper__(self):
        """
        Provides a quick summary and interpretation of HRV time-domain features

        References:
        [1] Yin Z, Liu C, Xie C, Nie Z, Wei J, Zhang W and Liang H (2025) Identification of atrial fibrillation using heart rate variability: a meta-analysis. Front. Cardiovasc. Med. 12:1581683. doi: 10.3389/fcvm.2025.1581683
        [2] H. Costin, C. Rotariu and A. Păsărică, "Atrial fibrillation onset prediction using variability of ECG signals," 2013 8TH INTERNATIONAL SYMPOSIUM ON ADVANCED TOPICS IN ELECTRICAL ENGINEERING (ATEE), Bucharest, Romania, 2013, pp. 1-4, doi: 10.1109/ATEE.2013.6563419. keywords: {Heart rate variability;Electrocardiography;Atrial fibrillation;Cardiology;Databases;Computers;Measurement;atrial fibrillation prediction;surface ECG;HRV analysis;morphologic variability;decision rule},
        [3] Anwar, A., & Khammari, H. An Efficient Paroxysmal Atrial Fibrillation Prediction Method Using CWT and SVM,
        [4] Buś, S., Jędrzejewski, K., & Guzik, P. (2022). Statistical and Diagnostic Properties of pRRx Parameters in Atrial Fibrillation Detection. Journal of Clinical Medicine, 11(19), 5702. https://doi.org/10.3390/jcm11195702, 
        [5] Parsi, A., Glavin, M., Jones, E., & Byrne, D. (2021). Prediction of paroxysmal atrial fibrillation using new heart rate variability features. Computers in biology and medicine, 133, 104367. https://doi.org/10.1016/j.compbiomed.2021.104367
        [6] Gavidia, M., Zhu, H., Montanari, A. N., Fuentes, J., Cheng, C., Dubner, S., Chames, M., Maison-Blanche, P., Rahman, M. M., Sassi, R., Badilini, F., Jiang, Y., Zhang, S., Zhang, H. T., Du, H., Teng, B., Yuan, Y., Wan, G., Tang, Z., He, X., … Goncalves, J. (2024). Early warning of atrial fibrillation using deep learning. Patterns (New York, N.Y.), 5(6), 100970. https://doi.org/10.1016/j.patter.2024.100970
        [7] C. Maier, M. Bauch and H. Dickhaus, "Screening and prediction of paroxysmal atrial fibrillation by analysis of heart rate variability parameters," Computers in Cardiology 2001. Vol.28 (Cat. No.01CH37287), Rotterdam, Netherlands, 2001, pp. 129-132, doi: 10.1109/CIC.2001.977608. keywords: {Atrial fibrillation;Heart rate variability;Testing;Electrocardiography;Performance analysis;Sampling methods;Heart rate detection;Heart rate;Fluctuations;Polynomials},


        """
        return {
            "HRV_MeanNN": "The mean of the RR intervals. [1-4]", 
            "HRV_SDNN": "The standard deviation of the RR intervals.[2]",
            "HRV_SDANN1, HRV_SDANN2, HRV_SDANN5[2] ": """The standard 
            deviation of average RR intervals extracted from n-minute 
            segments of time series data (1, 2 and 5 by default). 
            Note that these indices require a minimal duration of signal to be computed (3, 6 and 15 minutes respectively) 
            and will be silently skipped if the data provided is too short.""",
            "HRV_RMSSD": """The square root of the mean of the squared successive differences between  adjacent RR intervals.[1]""",
            "HRV_SDSD" : "The standard deviation of the successive differences between RR intervals.[7]",
            "HRV_CVNN" : "The standard deviation of the RR intervals (**SDNN**) divided by the mean of the RR intervals (**MeanNN**).[6]",
            "HRV_CVSD" : "The root mean square of successive differences (**RMSSD**) divided by the mean of the RR intervals (**MeanNN**).",
            "HRV_MedianNN": "The median of the RR intervals.",
            "HRV_MadNN": "The median absolute deviation of the RR intervals.",
            "HRV_MCVNN": "The median absolute deviation of the RR intervals (**MadNN**) divided by the median of the RR intervals (**MedianNN**).",
            "HRV_IQRNN": "The interquartile range (**IQR**) of the RR intervals.",
            "HRV_SDRMSSD": "SDNN / RMSSD, a time-domain equivalent for the low Frequency-to-High Frequency (LF/HF) Ratio (Sollers et al., 2007).",
            "HRV_Prc20NN": "The 20th percentile of the RR intervals ",
            "HRV_Prc80NN": "The 80th percentile of the RR intervals ",
            "HRV_pNN50": "The percentage of absolute differences in successive RR intervals greater than 50 ms [1]",
            "HRV_pNN20": "The percentage of absolute differences in successive RR intervals greater than 20 ms (Mietus et al., 2002).[1]",
            "HRV_MinNN": "The minimum of the RR intervals (Parent, 2019; Subramaniam, 2022).",
            "HRV_MaxNN": "The maximum of the RR intervals (Parent, 2019; Subramaniam, 2022).",
            "HRV_TINN": """A geometrical parameter of the HRV, or more specifically, the baseline width of
          the RR intervals distribution obtained by triangular interpolation, where the error of
          least squares determines the triangle. It is an approximation of the RR interval
          distribution.""",
            "HRV_HTI": """The HRV triangular index, measuring the total number of RR intervals divided by
          the height of the RR intervals histogram.""", 
            "NN50": """The number of pairs of successive R-R intervals that differ by more than 50 ms.[5]""",
            "pNN31": """The percentage of successive intervals differing by at least 31 ms. [4]""",
            "Mean_HR": "The mean heart rate in beats per minute (bpm).[1]",
            "SD_HR": "The standard deviation of the heart rate in bpm.[1]",
            "Min_HR": "The minimum heart rate in bpm.[1]",
            "Max_HR": "The maximum heart rate in bpm.[1]"
        }
    
