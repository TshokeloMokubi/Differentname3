import numpy as np
import pandas as pd

from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate
import surprise
import joblib
import pickle

train = pd.read_csv('~/unsupervised_data/unsupervised_movie_data/train.csv')
test = pd.read_csv('~/unsupervised_data/unsupervised_movie_data/test.csv')

# creating the model

cv = 3
train_subset = 0.1
random_state = 1
reader = surprise.Reader(rating_scale = (0.5,5.0))
dftrain = train.drop('timestamp', axis = 'columns')
dftrain = dftrain.sample(frac = train_subset, random_state = random_state)
dftrain = dftrain.reset_index(drop = True)
data = surprise.Dataset.load_from_df(dftrain, reader) 

param_grid = {'lr_all': np.arange(0.008,0.011,0.001), 'reg_all' : [0.1,0.3, 0.5]}
grid_s = surprise.model_selection.GridSearchCV(surprise.SVDpp,param_grid,measures = ['rmse','mae'],cv = cv)
grid_s.fit(data)

dict = grid_s.best_params['rmse']
dict

dftrain = train.drop('timestamp', axis = 'columns')
dftrain = dftrain.reset_index(drop = True)
data = surprise.Dataset.load_from_df(dftrain, reader) 

dict = grid_s.best_params['rmse']
alg = surprise.SVDpp(lr_all = dict['lr_all'], reg_all = dict['reg_all'])

output = surprise.model_selection.cross_validate(alg, data, verbose = True)

output['test_rmse']

dummies = [1]*len(test)
test['rating'] = dummies
predictions = alg.test(test.values)
del test['rating']

finpred = [ m.est for m in predictions]
test['rating'] = finpred


test.to_csv('jhb_rm61.csv', index = False)

with open('svdpp.pkl','wb') as file:
    pickle.dump(alg, file)








