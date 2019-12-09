# Machine Learning
Implementation of Machine Learning Algorithms for Time Series Forecasting

This project is a collection of machine learning algorithms that I implemented for a time series forecasting problem. Multilayer Perceptrons (MLP) and Long Short - Term Memory Networks are implemented. 

#Objective
To predict six-hourly (four times a day) values of storm surge height at a specified location by making use of neighboring oceanographic/atmospheric data such as wind speed and mean sea-level pressure.

#Dataset
Six hourly time series values of wind speed, mean sea-level pressure, and observed storm surge height at Cuxhaven (coastal city in Germany). A csv version of this dataset can be found in the same folder.

#Methodology
Due to the high autocorrelation in the dataset(both predictors and predictand), lagged predictors were used in the analysis. After preprocessing the data, MLP and LSTM models were trained and tested. A google collaboratory notebook for each implementation can be found in the same folder.

