# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 09:45:35 2021

@author: Nicos Evmides
"""
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
import get_and_pre_process_dataset

#to do: make yam config file instead of hard coding

model = MLPRegressor()
search_grid = {
     'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,1)],
     'activation': ['identity','relu','tanh','logistic'],
     'alpha': [0.0001, 0.05],
     'learning_rate': ['constant','invscaling','adaptive'],
     'solver': ['lbfgs','sgd','adam']
}

cv = 10
eval_metr = 'r2'
search = GridSearchCV(estimator=model,
                      param_grid=search_grid,
                      scoring=eval_metr,
                      cv=cv,n_jobs=-1,verbose=3)
x, y = get_and_pre_process_dataset.load_and_prepare_dataset(get_invalid_eta_as_null = False,eta_in_hours=True, "evmides")
search.fit(x, y)

print(search.best_params_)
print(search.best_score_)

# best results: trees =80, learning rate = 0.05
#'min_samples_split =1200, max_features = 'sqrt', subsample=0.8
#max_depth =10 or 15
