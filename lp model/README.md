## LP Model


### Done: 
	- Linear programing model is structured and implemented as a function.
	- Actual Load and PV Generation data of 31.12.2017 are used instead of forecasted ones.
	- Mostly hypotetical parameters are used.
	- Total cost minimization function works for one day (96 periods).
	- gurobipy solver module is used for linear optimization.

	
### Further works:
	- Some of these parameters are hypotetical and will be changed with true or reasonable values. 
	- Load and PV generation forecasts and Time-of-Use Tariff price parameters will be read from csv file.
	- Model function will be called for each period
\
