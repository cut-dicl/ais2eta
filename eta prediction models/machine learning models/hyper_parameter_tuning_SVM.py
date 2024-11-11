
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import get_and_pre_process_dataset

model = SVR(cache_size=2000)

search_grid = {
    #'degree': [2,3,4 ],
    #'gamma': ['scale', 'auto'],
    'kernel':['linear']   
    #'degree': [2,3,4],
    #'gamma': ['scale','auto'], 
    #'kernel':['linear','poly','rbf','sigmoid','precomputed']
    #'n_estimators': [60, 80, 100],
    #'learning_rate': [0.001, 0.05, 0.1, 0.15, 0.2],
    #'max_depth': [3, 5, 6, 10],
    #'min_child_weight': [1, 2],
    #'colsample_bytree': [0.5, 0.8],
    #'subsample': [0.5, 0.8]
}
cv = 10
eval_metr = 'r2'
search = GridSearchCV(estimator=model,
                      param_grid=search_grid,
                      scoring=eval_metr,
                      cv=cv,n_jobs=-1,verbose=3)
x, y = get_and_pre_process_dataset.load_and_prepare_dataset(get_invalid_eta_as_null = False,eta_in_hours=True)
search.fit(x, y)

print(search.best_params_)
print(search.best_score_)

