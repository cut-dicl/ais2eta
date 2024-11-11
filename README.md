# ais2eta
Employing AIS Data and Machine Learning to Enhance Prediction Accuracy of Vessel Arrival Times.

This project implements and tests several machine learning algorithms coded for predicting the Estimated Time of Arrival (ETA) of vessels based on AIS data.

## Dataset 

The dataset is contained in the file all_data_final.csv located in the `"dataset generation python code"` folder. The dataset contains 1 million AIS data points collected during July and August for vessels with portcalls at the main ports of Cyprus. Make sure to unzip the dataset before using it.

## Executables  

All the executable .py files are located in the directory `"eta prediction models/machine learning models/"`.

`ML_Models.py` contains a function for each model and the parameters used in each case. For example, the function get_dtr_model is a function used to initialize a decision tree model and will set all the parameters required for the model. There may be more than one functions for each algorithm. The difference is the parameters/setup which was suggested in other papers from the related literature. 

The parameters are read from the yml files in the directory `"eta prediction models/machine learning models/config files/"`. For example, for the get_dtr_model function, the parameters will be read from the file `optimized_dtr_hyperparms.yml`.

### Finding Optimal Hyperparameters

In the executables folder, you will find .py files with the following naming convention: `hyper_parameter_tuning_<model name>.py`
Example: hyper_parameter_tuning_DTR.py

The python file near the beginning contains a list with parameter names and values separated by comas (in brackets).

For example, for Decision Tree (DTR):
```
  'splitter' : ['best', 'random'],
  'max_depth' : [1,5,15,20,25],
  'min_samples_split' : [10,50,100,136, 150,200],
  'min_samples_leaf' : [100, 1000, 2000, 2221, 3000],
  'min_weight_fraction_leaf' : [0.0, 0.1, 0.2],
```

These are the parameter values to be tested using Grid Search to find the optimal combination of values. You can execute the code using the command (for DTR):
```
python hyper_parameter_tuning_DTR.py
```

The optimal parameter must then be entered in the corresponding yml file.

### Cross Validation Testing

Once the parameters (or features) are in place, then we need to run the cross validation by issuing the following command (for DTR):
```
python cross_validation_trials_dtr.py
```

This will generate graphic charts and csv files with the results.


References:
-----------
N. Evmides, S. Aslam, T. Ramez, M. Michaelides, and H. Herodotou. Enhancing Prediction Accuracy of Vessel Arrival Times Using Machine Learning. Journal of Marine Science and Engineering (JMSE), Vol. 12, No. 8, Article 1362, 17 pages, August 2024.

Authors:
--------
Nicos Evmides <br />
Cyprus University of Technology, Cyprus <br />
Email: nicos.evmides@cut.ac.cy <br />

Sheraz Aslam (PhD., Member IEEE) <br />
Cyprus University of Technology, Cyprus <br />

Tzioyntmprian T. Ramez <br />
Cyprus University of Technology, Cyprus <br />

Michalis Michaelides <br />
Cyprus University of Technology, Cyprus <br />

Herodotos Herodotou <br />
Cyprus University of Technology, Cyprus <br />
https://dicl.cut.ac.cy/
