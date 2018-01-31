## LP Model


### Done: 
- Linear programming model is structured and implemented as a function.
- Actual Load and PV Generation data of 31.12.2017 are used instead of forecasted ones.
- Mostly hypothetical parameters are used.
- Total cost minimization function works for one day (96 periods).
- gurobipy solver module is used for linear optimization.
- Load and PV generation forecasts and Time-of-Use Tariff price parameters are read from csv file.

	
### Further works:
- Some of these parameters are hypothetical and will be changed with true or reasonable values. 
- Model function will be called continuously for each period.

\
