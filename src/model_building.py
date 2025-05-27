import numpy as np
import pandas as pd
import pickle
import yaml

from sklearn.ensemble import GradientBoostingClassifier

params = yaml.safe_load(open('params.yaml','r'))['model_building']

#fretch data from data/processed
train_data = pd.read_csv('./data/features/train_bow.csv')


x_train = train_data.iloc[:,0:-1].values
y_train = train_data.iloc[:,-1].values

#define and train the GradientBoostingClassifier

clf = GradientBoostingClassifier(n_estimators=params['n_estimators'], learning_rate=params['learning_rate'])
clf.fit(x_train,y_train)

#save
pickle.dump(clf, open('model.pkl','wb'))