import sys
import os

import numpy as np
import matplotlib.pyplot as plt

from stochasticModelFrame import stochasticModelFrame
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))
import utilFunctions as UF


fs, x = UF.wavread('../../../sounds/ocean.wav')
w = np.hanning(1024)
N = 1024
stocf = 0.1
maxFreq = 10000.0
lastbin = N * maxFreq / fs
first = 1000
last = first + w.size
mX, pX, mY, pY, y = stochasticModelFrame(x[first:last], w, N, stocf)

plt.figure(1, figsize=(9, 5))
plt.subplot(3, 1, 1)
plt.plot(float(fs) * np.arange(mY.size) / N, mY, 'r', lw=1.5, label="mY")
plt.axis([0, maxFreq, -78, max(mX) + 0.5])
plt.title('mY (stochastic approximation of mX)')
plt.subplot(3, 1, 2)
plt.plot(float(fs) * np.arange(pY.size) / N, pY - np.pi, 'c', lw=1.5, label="pY")
plt.axis([0, maxFreq, -np.pi, np.pi])
plt.title('pY (random phases)')
plt.subplot(3, 1, 3)
plt.plot(np.arange(first, last) / float(fs), y, 'b', lw=1.5)
plt.axis([first / float(fs), last / float(fs), min(y), max(y)])
plt.title('yst')

plt.tight_layout()
plt.savefig('stochasticSynthesisFrame.png')
plt.show()
