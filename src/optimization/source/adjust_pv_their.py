import pandas as pd

df = pd.read_csv('../result/their_forecast/g_mip_battery_pv.csv',index_col=0)
df.insert(7, 'battery', df['battery-out']-df['battery-in'])
df.insert(7, 'grid_forecast', df['grid']-df['feed'])
df.drop(['battery-out','battery-in','grid','feed'],axis=1, inplace=True)
price = pd.read_csv('../data/price.csv')
price = pd.concat([price]*360,ignore_index=True)
df.insert(10, 'price', price['germany'])


##delta gridi bul
df1=df[df['pv_actual']+df['battery']>0]
df1['delta_grid']=df1['load_actual']-df1['grid_forecast']-0.9*(df1['pv_actual']+df1['battery'])
df2=df[df['pv_actual']+df['battery']<=0]
df2['delta_grid']=df2['load_actual']-df2['grid_forecast']-(1/0.9)*(df2['pv_actual']+df2['battery'])
df=pd.concat([df1,df2]).sort_index()
df.insert(7, 'grid_actual', df['delta_grid']+df['grid_forecast'])
df.drop(['delta_grid'],axis=1, inplace=True)

##cost hesapla
df1=df[df['grid_actual']>0]
df1['adjusted_cost']=df1['grid_actual']*df1['price']*0.25
df2=df[df['grid_actual']<=0]
df2['adjusted_cost']=df2['grid_actual']*df2['price']*0.25*0.9
df=pd.concat([df1,df2]).sort_index()

df.to_csv('../result/their_adjusted/g_mip_battery_pv.csv')

########

df = pd.read_csv('../result/their_adjusted/g_mip_battery_pv.csv', index_col=0)
costdf = pd.read_csv('../result/their_forecast/g_mip_battery_pv_cost.csv', index_col=0)
daily_cost = []
for x in range(360):
    daily_cost.append(df.iloc[96*x : 96*(x+1)]['adjusted_cost'].sum())
print(costdf['daily_cost'].sum())
print(sum(daily_cost))
costdf['adjusted_daily_cost'] = pd.Series(daily_cost)
costdf.to_csv('../result/their_adjusted/g_mip_battery_pv_cost.csv')

########