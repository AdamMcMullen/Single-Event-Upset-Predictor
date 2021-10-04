import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

"""
Author: Adam McMullen
Team: Team NorthStar
Motto: Sic Itur Ad Astra
Date: October 3, 2021

This script extracts the CASSIOPE, Kp, DST, and F10.7 data. These are cleaned and then combines these all into a pandas
dataframe that will be analyzed with machine learning
"""

SEU=pd.read_csv("SEU.csv")
non=pd.read_csv("non.csv")

SEU['class']=['SEU']*len(SEU)
non['class']=['Non']*len(non)

try:
    dataset=pd.read_pickle("dataset.pkl")
    test=pd.read_pickle("test.pkl")
except:
    dataset=pd.concat([SEU,non])
    test = pd.read_csv("test.csv")

    def clean_time(dataset):
        # .astype('int64')//1e9
        dataset['#Time'] = pd.to_datetime(dataset['#Time'], format='%d/%m/%Y %H:%M')
        dataset['Hour'] = pd.to_datetime(dataset['#Time']).dt.hour
        dataset['Min'] = pd.to_datetime(dataset['#Time']).dt.minute
        dataset['Date'] = pd.to_datetime(dataset['#Time']).dt.date.apply(datetime.toordinal)
        dataset['Year'] = pd.to_datetime(dataset['#Time']).dt.year
        dataset['Month'] = pd.to_datetime(dataset['#Time']).dt.month
        dataset['Day'] = pd.to_datetime(dataset['#Time']).dt.day
        dataset['DayOfYear'] = pd.to_datetime(dataset['#Time']).dt.dayofyear
        dataset['LocalTime'] = (dataset['Hour'] + dataset['Min'] / 60. + dataset[' Longitude (deg)'] / 15.) % 24
        return dataset

    def get_kp(year, month, day, hour,min):
        # from here: ftp://ftp.swpc.noaa.gov/pub/indices/old_indices/
        # info: https: // www.swpc.noaa.gov / sites / default / files / images / u2 / TheK - index.pdf
        f = open("kp/"+str(year)+"_DGD.txt","r")
        kp=-1
        for row in f:
            if not row[0]==":" and not row[0]=="#":
                fields = row.replace("\n","").split('   ')
                date = fields[0].split(' ')
                if int(date[0])==year and int(date[1])==month and int(date[2])==day:
                    kp = list(filter(None, fields[3].split(' ')))[1:][int((hour+min/60)//3)]
                    break
        f.close()
        return kp


    def get_dst(year, dayofyear, hour):
        # from here: https: // spdf.gsfc.nasa.gov / pub / data / omni / high_res_omni / monthly_1min /
        # data explination: https://omniweb.gsfc.nasa.gov/html/ow_data.html
        f = open("DST/omni2_" + str(year) + ".dat", "r")
        dst = -1
        for row in f:
            fields = list(filter(None, row.replace("\n",'').split(' ')))
            if int(fields[0]) == year and int(fields[1]) == dayofyear and int(fields[2]) == hour:
                dst = fields[40]
                break
        f.close()
        return dst

    def get_f10(year,month,day,hour,min):
        # from here: ftp://ftp.seismo.nrcan.gc.ca/spaceweather/solar_flux/daily_flux_values/
        f = open("F10/fluxtable.txt", "r")
        f10 = -1
        dhour=24
        for row in f:
            fields = list(filter(None, row.replace("\n", '').split(' ')))
            if fields[0] == str(year)+str(month).zfill(2)+str(day).zfill(2):
                if np.abs(int(fields[1][:2])-hour-min/60) < dhour:
                    dhour = np.abs(int(fields[1][:2]) - hour)
                    f10 = fields[5]
                    break
        f.close()
        return float(f10)

    def get_space_weather(dataset):
        kp=np.zeros(dataset.shape[0])
        dst=np.zeros(dataset.shape[0])
        f10=np.zeros(dataset.shape[0])
        SAA=np.empty(dataset.shape[0])

        for i in np.arange(dataset.shape[0]):
            print(i)
            SAA[i]=True if -50 < dataset[' Latitude (deg)'].values[i] < 0  and -90 < dataset[' Longitude (deg)'].values[i] < 40 else False
            kp[i]=get_kp(dataset['Year'].values[i], dataset['Month'].values[i], dataset['Day'].values[i], dataset['Hour'].values[i],dataset['Min'].values[i])
            dst[i]=get_dst(dataset['Year'].values[i], dataset['DayOfYear'].values[i], dataset['Hour'].values[i])
            f10[i]=get_f10(dataset['Year'].values[i], dataset['Month'].values[i], dataset['Day'].values[i], dataset['Hour'].values[i],dataset['Min'].values[i])

        dataset['SAA'] = SAA
        dataset['kp']=kp
        dataset['dst']=dst
        dataset['f10']=f10
        return dataset


    dataset = clean_time(dataset)
    test = clean_time(test)

    dataset=get_space_weather(dataset)
    test = get_space_weather(test)

    dataset.to_pickle("dataset.pkl")
    test.to_pickle("test.pkl")