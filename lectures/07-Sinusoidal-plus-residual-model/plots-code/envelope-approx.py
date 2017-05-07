import sys
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))
import utilFunctions as UF
import dftModel as DFT


fs, x = UF.wavread('../../../sounds/ocean.wav')
w = np.hanning(1024)
N = 1024
stocf = .1
envSize = (N * stocf) // 2
maxFreq = 10000.0
lastbin = N * maxFreq / fs
first = 4000
last = first + w.size
mX, pX = DFT.dftAnal(x[first:last], w, N)
mXenv = resample(np.maximum(-200, mX), envSize)
mY = resample(mXenv, N / 2)

plt.figure(1, figsize=(9, 5))
plt.plot(float(fs) * np.arange(mX.size) / N, mX, 'r', lw=1.5, label=r'$a$')
plt.plot(float(fs / 2.0) * np.arange(0, envSize) / envSize, mXenv, color='k', lw=1.5, label=r'$\tilde a$')
plt.plot(float(fs) * np.arange(0, N / 2) / N, mY, 'g', lw=1.5, label=r'$b$')
plt.legend(fontsize=18)
plt.axis([0, maxFreq, -75, max(mX)])
plt.title('envelope approximation')

plt.tight_layout()
plt.savefig('envelope-approx.png')
plt.show()
