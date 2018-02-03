## LP Model


### Done: 
- Linear programming model is structured and implemented as a function.
- Actual Load and Actual PV Generation data from 01.01.2012 to 31.12.2017 are used instead of forecasted ones.
- Actual Load, Actual PV Generation and Time-of-Use Tariff prices parameters are read from csv file.
- Total cost minimization function works for one day (96 periods) and can be called for each period continuously.
- gurobipy solver module is used for linear optimization.

	
### Further works:
- Most of the parameters are hypothetical and these parameters will be changed with true or reasonable values.
- Actual Load and Actual PV Generation data will be replaced by Load and PV Generation forecast data.

