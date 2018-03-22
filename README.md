Make sure to have keras, tensorflow, pandas, numpy, matplotlib, and sklearn installed.

To get data on S&P500, go to https://finance.yahoo.com/quote/%5EGSPC/history?p=%5EGSPC,
select the maximum date range, and download the .csv file.

To run the model, run 'python run_lstm_prediction.py' in terminal.
	- uncomment the line for the corresponding prediction that you wish to run

To run the hyperparameter search, run 'python hyperparameter_tuning.py'
	- comment out the lines that aren't the hyperparameter that you wish to tune.

The built LSTM models are located in 'build_lstm.py'

The values picked from hyperparameter tuning are in 'chosen_hyperparameters.py'

Lots of data files are in /data

Method for running training and prediction resides in 'keras_impl_lstm.py'

Methods for plotting the graphs of predictions resides in 'plotting_methods.py'

Methods for preprocessing data reside in 'preprocess_data.py'
