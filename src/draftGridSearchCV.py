# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.grid_search import GridSearchCV

# <codecell>

#donations = pd.read_csv('../data/donations.csv').sort('projectid')
projects = pd.read_csv('../data/projects.csv').sort('projectid')
outcomes = pd.read_csv('../data/outcomes.csv').sort('projectid')
#resources = pd.read_csv('../data/resources.csv').sort('projectid')
sample = pd.read_csv('../data/sampleSubmission.csv').sort('projectid')
#essays = pd.read_csv('../data/essays.csv').sort('projectid')

# <codecell>

dates = np.array(projects.date_posted)
train_idx = np.where(dates < '2014-01-01')[0]
test_idx = np.where(dates >= '2014-01-01')[0]

# <codecell>

projects = projects.fillna(method='pad')

# <codecell>

outcomes = np.array(outcomes.is_exciting)

# <codecell>

projectCatogorialColumns = ['school_city', 'school_state', 'school_zip', 'school_metro', 'school_district', 'school_county', 'school_charter', 'school_magnet',
 'school_year_round', 'school_nlns', 'school_kipp', 'school_charter_ready_promise', 'teacher_prefix', 'teacher_teach_for_america', 'teacher_ny_teaching_fellow', 'primary_focus_subject','primary_focus_area', 
'secondary_focus_subject', 'secondary_focus_area', 'resource_type', 'poverty_level', 'grade_level',
'students_reached', 'eligible_double_your_impact_match', 'eligible_almost_home_match' ]

# <codecell>

projects = np.array(projects[projectCatogorialColumns])

# <codecell>

for i in range(0, projects.shape[1]):
    le = LabelEncoder()
    projects[:,i] = le.fit_transform(projects[:,i])
projects = projects.astype(float)

# <codecell>

ohe = OneHotEncoder()
projects = ohe.fit_transform(projects)

# <codecell>

train = projects[train_idx]
test = projects[test_idx]

# <codecell>

lr = LogisticRegression()
parameters = {'C':[0.01,0.1,1,10],'penalty':['l1','l2'] }
clf = GridSearchCV(lr, parameters, scoring = 'roc_auc', n_jobs = -1)

# <codecell>

clf.fit(train, outcomes=='t')

# <codecell>

clf.grid_scores_

# <codecell>

clf.best_estimator_

# <codecell>

clf.best_score_

# <codecell>

clf.best_params_

