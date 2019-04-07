import numpy as np

def s_channel_threshold(channel, threshold, binary):
    s_binary = np.zeros_like(channel)
    s_binary[(channel >= threshold[0]) & (channel <= threshold[1])] = 1

    combined_binary = np.zeros_like(binary)
    combined_binary[(s_binary == 1) | (binary == 1)] = 1
    return combined_binary