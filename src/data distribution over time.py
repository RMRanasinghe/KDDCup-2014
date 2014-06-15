# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pylab as plt

# <codecell>

projects = pd.read_csv('../data/projects.csv')
outcome = pd.read_csv('../data/outcomes.csv')

# <codecell>

projects = projects.merge(outcome, how = 'inner')

# <codecell>

projects = projects.sort('date_posted')

# <codecell>

projects.columns

# <codecell>

projects = projects[['date_posted','is_exciting','at_least_1_teacher_referred_donor', 'fully_funded','at_least_1_green_donation','great_chat',
'three_or_more_non_teacher_referred_donors', 'one_non_teacher_referred_donor_giving_100_plus','donation_from_thoughtful_donor']]

# <codecell>

le = LabelEncoder()

# <codecell>

for i in range(0,projects.shape[0]):
    projects.date_posted[i] = projects.date_posted[i][0:7]

# <codecell>

projects['date_posted'] = le.fit_transform(projects.date_posted)

# <codecell>

plotNumb = 1
for col in projects.columns:
    if(col != 'date_posted'):
        tot = projects[projects.date_posted == 0]
        true = [float(tot[tot[col] == 't'].count()[0])/float(tot.count()[0])]
    
        for i in range(1,projects.date_posted.max()+1):
            tot = projects[projects.date_posted == i]
            true.append(float(tot[tot[col] == 't'].count()[0])/float(tot.count()[0]))
        
        plt.subplot(projects.shape[1]/2, 2, plotNumb)
        plt.plot(true,'b.-')
        plt.title(col)
        plotNumb = plotNumb + 1

# <codecell>

plt.show()

# <codecell>


