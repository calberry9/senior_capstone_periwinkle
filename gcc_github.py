import numpy as np
import matplotlib.pyplot as plt

def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=16):
    '''
    This function computes the offset between the signal sig and the reference signal refsig
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT)method.
    '''
    
    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    n = sig.shape[0] + refsig.shape[0]

    # Generalized Cross Correlation Phase Transform
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)

    cc = np.fft.irfft(R / np.abs(R), n=(interp * n))

    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))

    # find max cross correlation index
    shift = np.argmax(np.abs(cc)) - max_shift

    tau = shift / float(interp * fs)
    
    return shift, cc


def main():
    
    refsig = np.linspace(1, 10, 10)

    for i in range(0, 10):
        sig = np.concatenate((np.linspace(0, 0, i), refsig, np.linspace(0, 0, 10 - i)))
        offset, _ = gcc_phat(sig, refsig)
        print(offset)


if __name__ == "__main__":
    fs = 16000
    chunk_size = 1024
    T = 1024/fs
    t = np.linspace(0.0, T, chunk_size)
    w = 2*np.pi*100
    delay_time = 1e-3
    delay_samples = int(delay_time*fs)
    print(delay_samples)

    refsig = np.sin(w*t)
    delay = np.zeros(delay_samples)
    sig = np.concatenate([delay, refsig])
    sig = sig[0:chunk_size]

    fig1, axs1 = plt.subplots(2, 1)
    axs1[0].plot(t, refsig)
    axs1[1].plot(t, sig)
    plt.show()
    
    tau, cc = gcc_phat(sig, refsig, fs=1, max_tau=None, interp=16)
    
    plt.plot(cc)
    plt.show()
    
    print(tau)
    