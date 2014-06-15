# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import pandas as pd

# <codecell>

import numpy as np

# <codecell>

df=pd.read_csv('KDDData/dataCSV/data/projects.csv')

# <codecell>

print df.columns

# <codecell>

totalCount = df.shape[0]  #total number of projects
for i in range(1,df.shape[1]):
    nullcount = df[df[df.columns[i]].isnull()].shape[0]   #null values
    percentage=str(float(nullcount)/float(totalCount) *100) + ' %'
    print df.columns[i],percentage

# <codecell>

totalCount = df.shape[0]
for i in range(1,df.shape[1]):
    nullcount = df[df[df.columns[i]].isnull()].shape[0]
    percentage=float(nullcount)/float(totalCount) *100
    if(percentage>1):
        print df.columns[i],percentage,'%'
        

# <codecell>


