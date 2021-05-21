import os
import pandas as pd
import numpy as np

def fixValues(df):
    # Replace '-' string with NaN
    df = df.replace ('-', '-1')

    # Also fixing ',' delimeter with '.' for float conversion '.' is for thousands , for the last delimeter
    # Casting str to float
    df['PM10 ( µg/m3 )'] = df['PM10 ( µg/m3 )'].astype(str).str.replace('.','')
    df['PM10 ( µg/m3 )'] = df['PM10 ( µg/m3 )'].astype(str).str.replace(',','.')
    df['PM10 ( µg/m3 )'] = pd.to_numeric(df['PM10 ( µg/m3 )'], downcast="float")
    
    df['SO2 ( µg/m3 )'] = df['SO2 ( µg/m3 )'].astype(str).str.replace('.','')
    df['SO2 ( µg/m3 )'] = df['SO2 ( µg/m3 )'].astype(str).str.replace(',','.')
    df['SO2 ( µg/m3 )'] = pd.to_numeric(df['SO2 ( µg/m3 )'], downcast="float")
    
    df['CO ( µg/m3 )'] = df['CO ( µg/m3 )'].astype(str).str.replace('.','')
    df['CO ( µg/m3 )'] = df['CO ( µg/m3 )'].astype(str).str.replace(',','.')
    df['CO ( µg/m3 )'] = pd.to_numeric(df['CO ( µg/m3 )'], downcast="float")
    
    df['NO2 ( µg/m3 )'] = df['NO2 ( µg/m3 )'].astype(str).str.replace('.','')
    df['NO2 ( µg/m3 )'] = df['NO2 ( µg/m3 )'].astype(str).str.replace(',','.')
    df['NO2 ( µg/m3 )'] = pd.to_numeric(df['NO2 ( µg/m3 )'], downcast="float")
    
    df['NOX ( µg/m3 )'] = df['NOX ( µg/m3 )'].astype(str).str.replace('.','')
    df['NOX ( µg/m3 )'] = df['NOX ( µg/m3 )'].astype(str).str.replace(',','.')
    df['NOX ( µg/m3 )'] = pd.to_numeric(df['NOX ( µg/m3 )'], downcast="float")
    
    df['O3 ( µg/m3 )'] = df['O3 ( µg/m3 )'].astype(str).str.replace('.','')
    df['O3 ( µg/m3 )'] = df['O3 ( µg/m3 )'].astype(str).str.replace(',','.')
    df['O3 ( µg/m3 )'] = pd.to_numeric(df['O3 ( µg/m3 )'], downcast="float")
    
    df['PM 2.5 ( µg/m3 )'] = df['PM 2.5 ( µg/m3 )'].astype(str).str.replace('.','')
    df['PM 2.5 ( µg/m3 )'] = df['PM 2.5 ( µg/m3 )'].astype(str).str.replace(',','.')
    df['PM 2.5 ( µg/m3 )'] = pd.to_numeric(df['PM 2.5 ( µg/m3 )'], downcast="float")
    
    return df
    
def fillEmptyRows(df):
    # Filling empty rows
    # https://towardsdatascience.com/7-ways-to-handle-missing-values-in-machine-learning-1a6326adf79e
    # 1- We cant just delete them because we need consistent timestamps
    # 2- If too many empty rows exists we should discard them
    # 3- Replacing missing data with mean/median
    # 3.1- This does not cover the covariance between features
    
    #df['PM10 ( µg/m3 )'] = df['PM10 ( µg/m3 )'].replace (-1.0, df[df != -1]['PM10 ( µg/m3 )'].median ())
    #df['SO2 ( µg/m3 )'] = df['SO2 ( µg/m3 )'].replace (-1.0, df[df != -1]['SO2 ( µg/m3 )'].median ())
    #df['CO ( µg/m3 )'] = df['CO ( µg/m3 )'].replace (-1.0, df[df != -1]['CO ( µg/m3 )'].median ())
    #df['NO2 ( µg/m3 )'] = df['NO2 ( µg/m3 )'].replace (-1.0, df[df != -1]['NO2 ( µg/m3 )'].median ())
    #df['NOX ( µg/m3 )'] = df['NOX ( µg/m3 )'].replace (-1.0, df[df != -1]['NOX ( µg/m3 )'].median ())
    #df['O3 ( µg/m3 )'] = df['O3 ( µg/m3 )'].replace (-1.0, df[df != -1]['O3 ( µg/m3 )'].median ())
    #df['PM 2.5 ( µg/m3 )'] = df['PM 2.5 ( µg/m3 )'].replace (-1.0, df[df != -1]['PM 2.5 ( µg/m3 )'].median ())
    if not(df['PM10 ( µg/m3 )'][df['PM10 ( µg/m3 )']!=-1].count()>0):df['PM10 ( µg/m3 )']=0
    if not(df['SO2 ( µg/m3 )'][df['SO2 ( µg/m3 )']!=-1].count()>0):df['SO2 ( µg/m3 )']=0
    if not(df['CO ( µg/m3 )'][df['CO ( µg/m3 )']!=-1].count()>0):df['CO ( µg/m3 )']=0
    if not(df['NO2 ( µg/m3 )'][df['NO2 ( µg/m3 )']!=-1].count()>0):df['NO2 ( µg/m3 )']=0
    if not(df['NOX ( µg/m3 )'][df['NOX ( µg/m3 )']!=-1].count()>0):df['NOX ( µg/m3 )']=0
    if not(df['O3 ( µg/m3 )'][df['O3 ( µg/m3 )']!=-1].count()>0):df['O3 ( µg/m3 )']=0
    if not(df['PM 2.5 ( µg/m3 )'][df['PM 2.5 ( µg/m3 )']!=-1].count()>0):df['PM 2.5 ( µg/m3 )']=0

    #if not(df['PM10 ( µg/m3 )'].notnull().values.any()): df['PM10 ( µg/m3 )']=0
    #if not(df['SO2 ( µg/m3 )'].notnull().values.any()): df['SO2 ( µg/m3 )']=0
    #if not(df['CO ( µg/m3 )'].notnull().values.any()): df['CO ( µg/m3 )']=0
    #if not(df['NO2 ( µg/m3 )'].notnull().values.any()): df['NO2 ( µg/m3 )']=0
    #if not(df['NOX ( µg/m3 )'].notnull().values.any()): df['NOX ( µg/m3 )']=0
    #if not(df['O3 ( µg/m3 )'].notnull().values.any()): df['O3 ( µg/m3 )']=0
    #if not(df['PM 2.5 ( µg/m3 )'].notnull().values.any()): df['PM 2.5 ( µg/m3 )']=0

    df = fillWithSameHourValue(df, 'PM10 ( µg/m3 )')
    df = fillWithSameHourValue(df, 'SO2 ( µg/m3 )')
    df = fillWithSameHourValue(df, 'CO ( µg/m3 )')
    df = fillWithSameHourValue(df, 'NO2 ( µg/m3 )')
    df = fillWithSameHourValue(df, 'NOX ( µg/m3 )')
    df = fillWithSameHourValue(df, 'O3 ( µg/m3 )')
    df = fillWithSameHourValue(df, 'PM 2.5 ( µg/m3 )')
    return df

def fillWithSameHourValue(df, pollutantColumnName):
    for x in range (df[pollutantColumnName].shape[0]):
        if(df[pollutantColumnName].iloc[x] == -1 and x < 24):
            df[pollutantColumnName].iloc[x] = findValue(df[pollutantColumnName], x)
        if(df[pollutantColumnName].iloc[x] == -1 and x >= 24):
            df[pollutantColumnName].iloc[x] = df[pollutantColumnName].iloc[x-24]
        
    return df

def findValue(df, i):
    if(df.shape[0]<=i):
        return 0

    if(df.iloc[i] == -1):
        i = i+24
        return findValue(df, i)
    else:
        return df.iloc[i] 

def preprocessing(df):
    df = fixValues(df)
    df = fillEmptyRows(df)
    return df