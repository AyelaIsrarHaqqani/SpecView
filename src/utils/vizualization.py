import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_sst_spectrum(signal, spectrum):
    plt.figure(figsize=(10,4))
    plt.plot(signal, label='Original')
    plt.plot(spectrum, label='SST Spectrum', alpha=0.7)
    plt.legend(); plt.title('SST Spectrum Visualization'); plt.show()