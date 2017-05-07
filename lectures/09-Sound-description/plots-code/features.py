
from soundAnalysis import plotFeatures


descInput = ('lowlevel.spectral_centroid.mean', 'lowlevel.mfcc.mean.2')
plt = plotFeatures('freesound-sounds', descInput=descInput, figure_index=1,
                   scatter_s=50, figure_kwargs={'figsize': (9.5, 6), })
plt.savefig('features.png')
