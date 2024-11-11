
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import get_and_pre_process_dataset

n_samples, n_features = 10, 5
rng = np.random.RandomState(0)
#y = rng.randn(n_samples)
#X = rng.randn(n_samples, n_features)
x, y = get_and_pre_process_dataset.load_and_prepare_dataset(get_invalid_eta_as_null = False,eta_in_hours=True)
regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
regr.fit(x, y)
Pipeline(steps=[('standardscaler', StandardScaler()),
                ('svr', SVR(epsilon=0.2))])

quit()

from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import get_and_pre_process_dataset

#X, y = make_regression(n_samples=200, random_state=1)
x, y = get_and_pre_process_dataset.load_and_prepare_dataset(get_invalid_eta_as_null = False,eta_in_hours=True)
X_train, X_test, y_train, y_test = train_test_split(x, y,random_state=1)
regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)
print(regr.predict(X_test[:2]))
#array([-0.9..., -7.1...])
print(regr.score(X_test, y_test))
quit()




from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import get_and_pre_process_dataset
from xgboost import XGBRegressor
#to do: make yam config file instead of hard coding
model = XGBRegressor()
search_grid = {
    'n_estimators': [80, 100],
    'learning_rate': [0.15,0.2],
    'max_depth': [5,6],
    'min_child_weight': [1],
    'colsample_bytree': [0.5],
    'subsample': [0.5]
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
