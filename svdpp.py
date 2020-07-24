from datetime import datetime
now = datetime.now().time() # time object
print("started running the script =", datetime.now().time())

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




print("finished loading the data =", datetime.now().time())
cv = 3
train_subset = 0.9
random_state = 1
reader = surprise.Reader(rating_scale = (0.5,5.0))
dftrain = train.drop('timestamp', axis = 'columns')
dftrain = dftrain.sample(frac = train_subset, random_state = random_state)
dftrain = dftrain.reset_index(drop = True)
data = surprise.Dataset.load_from_df(dftrain, reader) 
print("finished combining the data with the reader =", datetime.now().time())

# param_grid = {'lr_all': np.arange(0.008,0.011,0.001), 'reg_all' : [0.1,0.3, 0.5]}
# grid_s = surprise.model_selection.GridSearchCV(surprise.SVDpp,param_grid,measures = ['rmse','mae'],cv = cv)
# grid_s.fit(data)

# dict = grid_s.best_params['rmse']
# dict

# dftrain = train.drop('timestamp', axis = 'columns')
# dftrain = dftrain.reset_index(drop = True)
# data = surprise.Dataset.load_from_df(dftrain, reader) 
# dict = grid_s.best_params['rmse']

alg = surprise.SVDpp()#lr_all = dict['lr_all'], reg_all = dict['reg_all'])
print("finished creating the svdpp object =", datetime.now().time())

output = alg.fit(data.build_full_trainset())
print(output)
print("finished training the model =", datetime.now().time())


dummies = [1]*len(test)
test['rating'] = dummies
predictions = alg.test(test.values)
del test['rating']
print("finished predictions on testset =", datetime.now().time())
finpred = [ m.est for m in predictions]


zip2 = zip(test['userId'],test['movieId'])
li=[]
for o,p in zip2:
    li.append(str(o)+'_'+str(p))
finsub = pd.DataFrame(li, columns =  ['Id'])
finsub['rating'] = finpred

finsub.to_csv('jhb_rm61.csv', index = False)
print("finished saving a pickle =", datetime.now().time())
with open('svdpp.pkl','wb') as file:
    pickle.dump(alg, file)
print("finished saving a pickle and running the whole script =", datetime.now().time())






