# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 09:45:35 2021

@author: Nicos Evmides
"""
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
import get_and_pre_process_dataset

#to do: make yam config file instead of hard coding

model = DecisionTreeRegressor()
search_grid = {
    #'criterion' : ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
    'splitter' : ['best', 'random'],
    'max_depth' : [1,5,15,20,25],
    'min_samples_split' : [10,50,100,136, 150,200],
    'min_samples_leaf' : [100, 1000, 2000, 2221, 3000],
    'min_weight_fraction_leaf' : [0.0, 0.1, 0.2],
    #'max_features' : ['auto', 'sqrt', 'log2'],
    'max_features' : [1,2,3,4],
    'ccp_alpha' : [0.0,0.1]
}

cv = 10
eval_metr = 'r2'
search = GridSearchCV(estimator=model,
                      param_grid=search_grid,
                      scoring=eval_metr,
                      cv=cv,n_jobs=-1,verbose=3)
x, y, test_fold = get_and_pre_process_dataset.load_and_prepare_dataset(get_invalid_eta_as_null = False,eta_in_hours=True, paper="evmides")
search.fit(x, y)

print(search.best_params_)
print(search.best_score_)

# best results: trees =80, learning rate = 0.05
#'min_samples_split =1200, max_features = 'sqrt', subsample=0.8
#max_depth =10 or 15
