import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
"""
optimal = pd.read_csv("../result/optimal/t_mip_battery_cost.csv")
adjusted = pd.read_csv("../result/their_adjusted/t_mip_battery_cost.csv")
error = pd.Series(adjusted['adjusted_daily_cost']-optimal['daily_cost'])
plt.hist(error,60)
plt.xlabel('Daily Cost Error (TRY)')
plt.ylabel('Frequency')
print(error.mean(), error.std())
plt.text(0.0075, 150, r'$\mu=0.0018,\ \sigma=0.0030$')
plt.savefig("../result/figure/t_their_battery_optimal_adjusted_error.png")
plt.show()
plt.clf() # Mid
optimal = pd.read_csv("../result/optimal/t_mip_battery_pv_cost.csv")
adjusted = pd.read_csv("../result/their_adjusted/t_mip_battery_pv_cost.csv")
error = pd.Series(adjusted['adjusted_daily_cost']-optimal['daily_cost'])
plt.hist(error,65)
plt.xlabel('Daily Cost Error (TRY)')
plt.ylabel('Frequency')
print(error.mean(), error.std())
plt.text(0.025, 50, r'$\mu=0.0111,\ \sigma=0.0115$')
plt.savefig("../result/figure/t_their_battery_pv_optimal_adjusted_error.png")
plt.show()
"""
"""______________________________________________________________________________________________________________"""
"""
df = pd.read_csv("../data/pv.csv", index_col=0)
load_error_we = pd.Series(0.00036682367*(df["yhat0"]-df["actual"]))[96:34656]
load_error_they = pd.Series(0.00036682367*(df["forecast"]-df["actual"]))[96:34656]
plt.hist(load_error_we, 96)
plt.xlabel('PV Forecast Error (kWh)')
plt.ylabel('Frequency')
print(load_error_we.mean(), load_error_we.std())
plt.text(-0.35, 15000, r'$\mu=0.0036,\ \sigma=0.0709$')
plt.savefig("../result/figure/pv_forecast_error.png")
plt.show()
plt.clf()
plt.hist(load_error_they, 96)
plt.xlabel('PV Forecast Error (kWh)')
plt.ylabel('Frequency')
print(load_error_they.mean(), load_error_they.std())
plt.text(0.1, 15000, r'$\mu=0.0016,\ \sigma=0.0679$')
plt.savefig("../result/figure/their_pv_forecast_error.png")
plt.show()
"""
"""______________________________________________________________________________________________________________"""

df = pd.read_csv("../data/load.csv", index_col=0)
la = pd.Series(0.00001885071*df["actual"])[96:34656]
lw = pd.Series(0.00001885071*df["yhat0"])[96:34656]
lt = pd.Series(0.00001885071*df["forecast"])[96:34656]
lsqew = pd.Series((0.00001885071*df["yhat0"]-0.00001885071*df["actual"])**2)[96:34656]
lsqet = pd.Series((0.00001885071*df["forecast"]-0.00001885071*df["actual"])**2)[96:34656]
laew = pd.Series(0.00001885071*abs(df["yhat0"]-df["actual"]))[96:34656]
laet = pd.Series(0.00001885071*abs(df["forecast"]-df["actual"]))[96:34656]
lapew = pd.Series(0.001885071*abs((df["yhat0"]-df["actual"])/df["actual"]))[96:34656]
lapet = pd.Series(0.001885071*abs((df["forecast"]-df["actual"])/df["actual"]))[96:34656]
lapew.replace([np.inf, -np.inf], np.nan, inplace=True)
lapew.dropna(inplace=True)
lapet.replace([np.inf, -np.inf], np.nan, inplace=True)
lapet.dropna(inplace=True)
lmsew = lsqew.mean()
lmset = lsqet.mean()
lnmsew = lmsew*len(la)/(la.sum()*lw.sum())
lnmset = lmset*len(la)/(la.sum()*lt.sum())
lrmsew = lmsew**0.5
lrmset = lmset**0.5
lnrmsew = lrmsew/la.mean()
lnrmset = lrmset/la.mean()
lmaew = laew.mean()
lmaet = laet.mean()
lmapew = lapew.mean()
lmapet = lapet.mean()

df = pd.read_csv("../data/pv.csv", index_col=0)
pa = pd.Series(0.00036682367*df["actual"])[96:34656]
pw = pd.Series(0.00036682367*df["yhat0"])[96:34656]
pt = pd.Series(0.00036682367*df["forecast"])[96:34656]
psqew = pd.Series((0.00036682367*df["yhat0"]-0.00036682367*df["actual"])**2)[96:34656]
psqet = pd.Series((0.00036682367*df["forecast"]-0.00036682367*df["actual"])**2)[96:34656]
paew = pd.Series(0.00036682367*abs(df["yhat0"]-df["actual"]))[96:34656]
paet = pd.Series(0.00036682367*abs(df["forecast"]-df["actual"]))[96:34656]
papew = pd.Series(0.036682367*abs((df["yhat0"]-df["actual"])/df["actual"]))[96:34656]
papet = pd.Series(0.036682367*abs((df["forecast"]-df["actual"])/df["actual"]))[96:34656]
papew.replace([np.inf, -np.inf], np.nan, inplace=True)
papew.dropna(inplace=True)
papet.replace([np.inf, -np.inf], np.nan, inplace=True)
papet.dropna(inplace=True)
pmsew = psqew.mean()
pmset = psqet.mean()
pnmsew = pmsew*len(pa)/(pa.sum()**2)
pnmset = pmset*len(pa)/(pa.sum()**2)
prmsew = pmsew**0.5
prmset = pmset**0.5
pnrmsew = prmsew/pa.mean()
pnrmset = prmset/pa.mean()
pmaew = paew.mean()
pmaet = paet.mean()
pmapew = papew.mean()
pmapet = papet.mean()

ca = la-pa
cw = lw-pw
ct = lt-pt
csew = pd.Series((cw-ca)**2)
cset = pd.Series((ct-ca)**2)
caew = pd.Series(abs(cw-ca))
caet = pd.Series(abs(ct-ca))
capew = pd.Series(100*abs((cw-ca)/ca))
capet = pd.Series(100*abs((ct-ca)/ca))
cmsew = csew.mean()
cmset = cset.mean()
cnmsew = cmsew*len(ca)/(ca.sum()*cw.sum())
cnmset = cmset*len(ca)/(ca.sum()*ct.sum())
crmsew = cmsew**0.5
crmset = cmset**0.5
cnrmsew = crmsew/ca.mean()
cnrmset = crmset/ca.mean()
cmaew = caew.mean()
cmaet = caet.mean()
cmapew = capew.mean()
cmapet = capet.mean()


print("\n\nLOAD")
print("Mean = %12.12f\t\t\t Variance = %12.12f\t Standart Deviation = %12.12f" %(la.mean(), la.var(), la.std()))
print("Mean Squared Errors:\t\t\t We = %12.12f\t\t They = %12.12f" %(lmsew, lmset))
print("Root Mean Squared Errors:\t\t We = %12.12f\t\t They = %12.12f" %(lrmsew, lrmset))
print("Mean Absolute Errors:\t\t\t We = %12.12f\t\t They = %12.12f" %(lmaew, lmaet))
print("Normalized Mean Squared Error:\t\t We = %12.12f\t\t They = %12.12f" %(lnmsew, lnmset))
print("Normalized Root Mean Squared Errors:\t We = %12.12f\t\t They = %12.12f" %(lnrmsew, lnrmset))
print("Mean Absolute Percentage Errors (%%):\t We = %%%12.12f\t\t They = %%%12.12f\n" %(lmapew, lmapet))
print("PV")
print("Mean = %12.12f\t\t\t Variance = %12.12f\t Standart Deviation = %12.12f" %(pa.mean(), pa.var(), pa.std()))
print("Mean Squared Errors:\t\t\t We = %12.12f\t\t They = %12.12f" %(pmsew, pmset))
print("Root Mean Squared Errors:\t\t We = %12.12f\t\t They = %12.12f" %(prmsew, prmset))
print("Mean Absolute Errors:\t\t\t We = %12.12f\t\t They = %12.12f" %(pmaew, pmaet))
print("Normalized Mean Squared Error:\t\t We = %12.12f\t\t They = %12.12f" %(pnmsew, pnmset))
print("Normalized Root Mean Squared Errors:\t We = %12.12f\t\t They = %12.12f" %(pnrmsew, pnrmset))
print("Mean Absolute Percentage Errors (%%):\t We = %%%12.12f\t\t They = %%%12.12f\n" %(pmapew, pmapet))
print("\nCOMBINED (LOAD-PV)")
print("Mean = %12.12f\t\t\t Variance = %12.12f\t Standart Deviation = %12.12f" %(ca.mean(), ca.var(), ca.std()))
print("Mean Squared Errors:\t\t\t We = %12.12f\t\t They = %12.12f" %(cmsew, cmset))
print("Root Mean Squared Errors:\t\t We = %12.12f\t\t They = %12.12f" %(crmsew, crmset))
print("Mean Absolute Errors:\t\t\t We = %12.12f\t\t They = %12.12f" %(cmaew, cmaet))
print("Normalized Mean Squared Error:\t\t We = %12.12f\t\t They = %12.12f" %(cnmsew, cnmset))
print("Normalized Root Mean Squared Errors:\t We = %12.12f\t\t They = %12.12f" %(cnrmsew, cnrmset))
print("Mean Absolute Percentage Errors (%%):\t We = %%%12.12f\t\t They = %%%12.12f\n\n" %(cmapew, cmapet))
