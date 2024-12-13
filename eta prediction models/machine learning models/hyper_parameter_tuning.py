from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import get_and_pre_process_dataset
from xgboost import XGBRegressor
#to do: make yam config file instead of hard coding
model = XGBRegressor()
search_grid = {
    'n_estimators': [60, 80, 100],
    'learning_rate': [0.001, 0.05, 0.1, 0.15, 0.2],
    'max_depth': [3, 5, 6, 10],
    'min_child_weight': [1, 2],
    'colsample_bytree': [0.5, 0.8],
    'subsample': [0.5, 0.8]
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
