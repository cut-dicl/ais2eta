from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
import get_and_pre_process_dataset

#to do: make yam config file instead of hard coding

model = KNeighborsRegressor()
search_grid = {
    'n_neighbors': [4,5,6],
    'weights': ['uniform','distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree'],
    'leaf_size': [20,30,40],
    'p': [1,2]
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

# best results: trees =80, learning rate = 0.05
#'min_samples_split =1200, max_features = 'sqrt', subsample=0.8
#max_depth =10 or 15
