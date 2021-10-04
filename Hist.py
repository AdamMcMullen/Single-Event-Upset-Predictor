import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.neighbors import KernelDensity

from Data import *


"""
Author: Adam McMullen
Team: Team NorthStar
Motto: Sic Itur Ad Astra
Date: October 3, 2021

This script determines a probability of SEU using probability density functions constructed using past data.
"""

dataset = dataset[[' Latitude (deg)',' Longitude (deg)',' Altitude (km)','Year','LocalTime','kp','f10','dst','SAA','class']]
test = test[[' Latitude (deg)',' Longitude (deg)',' Altitude (km)','Year','LocalTime','kp','f10','dst','SAA']]

def get_probability(data,test):
    # KDE
    bw = 1.06*np.std(data)*len(data)**(-1/5)  # bandwidth
    kde = KernelDensity(bandwidth=bw, kernel='epanechnikov').fit(data[:, None])
    pdf_kde = np.exp(kde.score_samples(test[:, None]))

    # Hist
    num_bins = 10  # 100
    hist, histbins = np.histogram(data, bins=num_bins + 1, density=True)
    # find out what the pdf is for the data from the histogram
    ind = np.searchsorted(histbins, test)
    ind[ind == num_bins + 2] = num_bins + 1
    ind[ind == 0] = 1
    pdf_hist = hist[ind - 1]
    return np.asarray([pdf_hist,pdf_kde])

p_SEU=[1,1]
p_non=[1,1]

for ind in dataset.columns[:-1]:
    p_SEU*=get_probability(dataset[dataset['class']=='SEU'][ind].to_numpy(),test[ind].to_numpy()).T
    p_non*=get_probability(dataset[dataset['class']=='Non'][ind].to_numpy(),test[ind].to_numpy()).T

p_non*=1249/(22+1249)
p_SEU*=22/(22+1249)

norm=1/(p_SEU+p_non)

p_non*=norm*100
p_SEU*=norm*100

print('KDE')
print('The probability of the test events being SEUs is',[ '%.1f' % elem for elem in np.round(p_SEU[:,0],1) ],'%')
print('The probability of the test events being non-SEUs is',[ '%.1f' % elem for elem in np.round(p_non[:,0],1) ],'%')

print('\nHist')
print('The probability of the test events being SEUs is',[ '%.1f' % elem for elem in np.round(p_SEU[:,1],1) ],'%')
print('The probability of the test events being non-SEUs is',[ '%.1f' % elem for elem in np.round(p_non[:,1],1) ],'%')


# Results
# KDE
# The probability of the test events being SEUs is ['0.0', '95.2', '0.0', '37.8', '15.7', '0.0', '2.9', '0.0'] %
# The probability of the test events being non-SEUs is ['100.0', '4.8', '100.0', '62.2', '84.3', '100.0', '97.1', '100.0'] %
#
# Hist
# The probability of the test events being SEUs is ['5.6', '13.5', '1.2', '4.5', '5.1', '7.8', '0.0', '0.0'] %
# The probability of the test events being non-SEUs is ['94.4', '86.5', '98.8', '95.5', '94.9', '92.2', '100.0', '100.0'] %