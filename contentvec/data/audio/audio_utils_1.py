import os
import numpy as np
from scipy import signal as sg
import pdb


def make_lowshelf(g, fc, Q, fs=44100):
    """Generate filter coefficients for 2nd order Lowshelf filter.
    This function follows the code from the JUCE DSP library 
    which can be found in `juce_IIRFilter.cpp`. 
    
    The design equations are based upon those found in the Cookbook 
    formulae for audio equalizer biquad filter coefficients
    by Robert Bristow-Johnson. 
    https://www.w3.org/2011/audio/audio-eq-cookbook.html
    Args:
        g  (float): Gain factor in dB.
        fc (float): Cutoff frequency in Hz.
        Q  (float): Q factor.
        fs (float): Sampling frequency in Hz.
    Returns:
        tuple: (b, a) filter coefficients 
    """
    # convert gain from dB to linear
    g = np.power(10,(g/20))

    # initial values
    A = np.max([0.0, np.sqrt(g)])
    aminus1 = A - 1
    aplus1 = A + 1
    omega = (2 * np.pi * np.max([fc, 2.0])) / fs
    coso = np.cos(omega)
    beta = np.sin(omega) * np.sqrt(A) / Q 
    aminus1TimesCoso = aminus1 * coso

    # coefs calculation
    b0 = A * (aplus1 - aminus1TimesCoso + beta)
    b1 = A * 2 * (aminus1 - aplus1 * coso)
    b2 = A * (aplus1 - aminus1TimesCoso - beta)
    a0 = aplus1 + aminus1TimesCoso + beta
    a1 = -2 * (aminus1 + aplus1 * coso)
    a2 = aplus1 + aminus1TimesCoso - beta

    # output coefs 
    #b = np.array([b0/a0, b1/a0, b2/a0])
    #a = np.array([a0/a0, a1/a0, a2/a0])

    return np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])



def make_highself(g, fc, Q, fs=44100):
    """Generate filter coefficients for 2nd order Highshelf filter.
    This function follows the code from the JUCE DSP library 
    which can be found in `juce_IIRFilter.cpp`. 
    
    The design equations are based upon those found in the Cookbook 
    formulae for audio equalizer biquad filter coefficients
    by Robert Bristow-Johnson. 
    https://www.w3.org/2011/audio/audio-eq-cookbook.html
    Args:
        g  (float): Gain factor in dB.
        fc (float): Cutoff frequency in Hz.
        Q  (float): Q factor.
        fs (float): Sampling frequency in Hz.
    Returns:
        tuple: (b, a) filter coefficients 
    """
    # convert gain from dB to linear
    g = np.power(10,(g/20))

    # initial values
    A = np.max([0.0, np.sqrt(g)])
    aminus1 = A - 1
    aplus1 = A + 1
    omega = (2 * np.pi * np.max([fc, 2.0])) / fs
    coso = np.cos(omega)
    beta = np.sin(omega) * np.sqrt(A) / Q 
    aminus1TimesCoso = aminus1 * coso

    # coefs calculation
    b0 = A * (aplus1 + aminus1TimesCoso + beta)
    b1 = A * -2 * (aminus1 + aplus1 * coso)
    b2 = A * (aplus1 + aminus1TimesCoso - beta)
    a0 = aplus1 - aminus1TimesCoso + beta
    a1 = 2 * (aminus1 - aplus1 * coso)
    a2 = aplus1 - aminus1TimesCoso - beta

    # output coefs
    #b = np.array([b0/a0, b1/a0, b2/a0])
    #a = np.array([a0/a0, a1/a0, a2/a0])
      
    return np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])



def make_peaking(g, fc, Q, fs=44100):
    """Generate filter coefficients for 2nd order Peaking EQ.
    This function follows the code from the JUCE DSP library 
    which can be found in `juce_IIRFilter.cpp`. 
    
    The design equations are based upon those found in the Cookbook 
    formulae for audio equalizer biquad filter coefficients
    by Robert Bristow-Johnson. 
    https://www.w3.org/2011/audio/audio-eq-cookbook.html
    Args:
        g  (float): Gain factor in dB.
        fc (float): Cutoff frequency in Hz.
        Q  (float): Q factor.
        fs (float): Sampling frequency in Hz.
    Returns:
        tuple: (b, a) filter coefficients 
    """
    # convert gain from dB to linear
    g = np.power(10,(g/20))

    # initial values
    A = np.max([0.0, np.sqrt(g)])
    omega = (2 * np.pi * np.max([fc, 2.0])) / fs
    alpha = np.sin(omega) / (Q * 2)
    c2 = -2 * np.cos(omega)
    alphaTimesA = alpha * A
    alphaOverA = alpha / A

    # coefs calculation
    b0 = 1 + alphaTimesA
    b1 = c2
    b2 = 1 - alphaTimesA
    a0 = 1 + alphaOverA
    a1 = c2
    a2 = 1 - alphaOverA

    # output coefs
    #b = np.array([b0/a0, b1/a0, b2/a0])
    #a = np.array([a0/a0, a1/a0, a2/a0])
    
    return np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])



def params2sos(G, Fc, Q, fs):
    """Convert 5 band EQ paramaters to 2nd order sections.
    Takes a vector with shape (13,) of denormalized EQ parameters
    and calculates filter coefficients for each of the 5 filters.
    These coefficients (2nd order sections) are then stored into a
    single (5,6) matrix. This matrix can be fed to `scipy.signal.sosfreqz()`
    in order to determine the frequency response of the cascasd of
    all five biquad filters.
    Args:
        x  (float): Gain factor in dB.       
        fs (float): Sampling frequency in Hz.
    Returns:
        ndarray: filter coefficients for 5 band EQ stored in (5,6) matrix.
        [[b1_0, b1_1, b1_2, a1_0, a1_1, a1_2],  # lowshelf coefficients
         [b2_0, b2_1, b2_2, a2_0, a2_1, a2_2],  # first band coefficients
         [b3_0, b3_1, b3_2, a3_0, a3_1, a3_2],  # second band coefficients
         [b4_0, b4_1, b4_2, a4_0, a4_1, a4_2],  # third band coefficients
         [b5_0, b5_1, b5_2, a5_0, a5_1, a5_2]]  # highshelf coefficients
    """
    # generate filter coefficients from eq params
    c0 = make_lowshelf(G[0], Fc[0], Q[0], fs=fs)
    c1 = make_peaking (G[1], Fc[1], Q[1], fs=fs)
    c2 = make_peaking (G[2], Fc[2], Q[2], fs=fs)
    c3 = make_peaking (G[3], Fc[3], Q[3], fs=fs)
    c4 = make_peaking (G[4], Fc[4], Q[4], fs=fs)
    c5 = make_peaking (G[5], Fc[5], Q[5], fs=fs)
    c6 = make_peaking (G[6], Fc[6], Q[6], fs=fs)
    c7 = make_peaking (G[7], Fc[7], Q[7], fs=fs)
    c8 = make_peaking (G[8], Fc[8], Q[8], fs=fs)
    c9 = make_highself(G[9], Fc[9], Q[9], fs=fs)

    # stuff coefficients into second order sections structure
    sos = np.concatenate([c0,c1,c2,c3,c4,c5,c6,c7,c8,c9], axis=0)

    return sos


import parselmouth
def change_gender(x, fs, lo, hi, ratio_fs, ratio_ps, ratio_pr):
    s = parselmouth.Sound(x, sampling_frequency=fs)
    f0 = s.to_pitch_ac(pitch_floor=lo, pitch_ceiling=hi, time_step=0.8/lo)
    f0_np = f0.selected_array['frequency']
    f0_med = np.median(f0_np[f0_np!=0]).item()
    ss = parselmouth.praat.call([s, f0], "Change gender", ratio_fs, f0_med*ratio_ps, ratio_pr, 1.0)
    return ss.values.squeeze(0)

def change_gender_f0(x, fs, lo, hi, ratio_fs, new_f0_med, ratio_pr):
    s = parselmouth.Sound(x, sampling_frequency=fs)
    ss = parselmouth.praat.call(s, "Change gender", lo, hi, ratio_fs, new_f0_med, ratio_pr, 1.0)
    return ss.values.squeeze(0)

