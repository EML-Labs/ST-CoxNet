import numpy as np

class Alpha1:
    def __init__(self):
        pass

    def compute(self, rr_window) -> float:
        """
        Calculates the short-term scaling exponent (alpha 1) using DFA.
        A reduction implies a change to a more random state[cite: 259].
        """
        N = len(rr_window)
        # 1. Integrate the signal [cite: 61]
        y = np.cumsum(rr_window - np.mean(rr_window))
        
        # 2. Define scales for short-term (4 to 11 beats) 
        scales = np.arange(4, 12)
        fluctuations = []
        
        for n in scales:
            # Divide into segments of length n
            segments = N // n
            rms_n = []
            for i in range(segments):
                segment = y[i*n : (i+1)*n]
                x_axis = np.arange(n)
                # Fit a linear trend (detrending) [cite: 61]
                poly = np.polyfit(x_axis, segment, 1)
                trend = np.polyval(poly, x_axis)
                # Calculate RMS of the detrended segment
                rms_n.append(np.sqrt(np.mean((segment - trend)**2)))
            fluctuations.append(np.mean(rms_n))
        
        # 3. Calculate the slope of log(scale) vs log(fluctuation)
        alpha1 = np.polyfit(np.log(scales), np.log(fluctuations), 1)[0]
        return alpha1

# def calculate_alpha1(self, rr_window):
#     """
#     Calculates the short-term scaling exponent (alpha 1) using DFA.
#     A reduction implies a change to a more random state[cite: 259].
#     """
#     N = len(rr_window)
#     # 1. Integrate the signal [cite: 61]
#     y = np.cumsum(rr_window - np.mean(rr_window))
    
#     # 2. Define scales for short-term (4 to 11 beats) 
#     scales = np.arange(4, 12)
#     fluctuations = []
    
#     for n in scales:
#         # Divide into segments of length n
#         segments = N // n
#         rms_n = []
#         for i in range(segments):
#             segment = y[i*n : (i+1)*n]
#             x_axis = np.arange(n)
#             # Fit a linear trend (detrending) [cite: 61]
#             poly = np.polyfit(x_axis, segment, 1)
#             trend = np.polyval(poly, x_axis)
#             # Calculate RMS of the detrended segment
#             rms_n.append(np.sqrt(np.mean((segment - trend)**2)))
#         fluctuations.append(np.mean(rms_n))
    
#     # 3. Calculate the slope of log(scale) vs log(fluctuation)
#     alpha1 = np.polyfit(np.log(scales), np.log(fluctuations), 1)[0]
#     return alpha1