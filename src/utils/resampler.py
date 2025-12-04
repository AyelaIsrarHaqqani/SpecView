from scipy.signal import resample

def resample_signal(signal, length=256):
    return resample(signal, length)
