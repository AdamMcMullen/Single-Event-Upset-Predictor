import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from Data import *


"""
Author: Adam McMullen
Team: Team NorthStar
Motto: Sic Itur Ad Astra
Date: October 3, 2021

This script plots the various single event upset data to identify relationships between SEUs, orbital and solar weather
variables.
"""

dataset = dataset[[' Latitude (deg)',' Longitude (deg)',' Altitude (km)','Year','LocalTime','kp','f10','dst','SAA','class']]

def summary(dataset):
    print(dataset.shape)
    print(dataset.columns)
    print(dataset.head(20))
    print(dataset.describe())
    print(dataset.groupby('class').size())


def plotter(dataset):
    # box and whisker plots
    dataset.plot(kind='box', subplots=True, layout=(3, 3), sharex=False, sharey=False)
    plt.show()

    # histograms
    dataset.hist()
    plt.show()

    # scatter plot matrix
    pd.plotting.scatter_matrix(dataset)
    plt.show()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="mollweide")
ax.plot(dataset[dataset['class']=='Non'][' Longitude (deg)']*np.pi/180,dataset[dataset['class']=='Non'][' Latitude (deg)']*np.pi/180,'k.')
ax.plot(dataset[dataset['class']=='SEU'][' Longitude (deg)']*np.pi/180,dataset[dataset['class']=='SEU'][' Latitude (deg)']*np.pi/180,'r.',markersize=5)
ax.plot(test[' Longitude (deg)']*np.pi/180,test[' Latitude (deg)']*np.pi/180,'*',markersize=10)
ax.grid()
plt.show()

summary(dataset)
summary(SEU)
summary(non)
summary(test)

plotter(dataset)
plotter(SEU)
plotter(non)
plotter(test)

weights=np.append(np.ones(len(SEU))/len(SEU),np.ones(len(non))/len(non))
dataset['weights']=weights
# scatter plot matrix
dataset= dataset.loc[~dataset.index.duplicated(), :]
print(dataset.columns)

for ind in dataset.columns[:-2]:
    sns.kdeplot(data=dataset, x=ind, hue="class", fill=True,common_norm=False,clip=[np.min(dataset[ind]),np.max(dataset[ind])])
    plt.show()

sns.displot(dataset, x="SAA", hue="class",  fill=True, weights=dataset['weights'])
plt.show()

sns.pairplot(dataset,hue='class',markers='.')
plt.show()


