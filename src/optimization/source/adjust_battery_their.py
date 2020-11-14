import pandas as pd


df = pd.read_csv('../result/their_forecast/g_mip_battery.csv', index_col=0)
price = pd.read_csv('../data/price.csv')
price = pd.concat([price]*360,ignore_index=True)
df.insert(10, 'price', price['germany'])

part1 = df[df['load_actual']>df['load_forecast']]
part1['adjusted_cost'] = part1['cost'] + (part1['load_actual']-part1['load_forecast'])*part1['price']*0.25

part2 = df[df['load_actual']<=df['load_forecast']]

part2p = part2[(part2['load_forecast']-part2['load_actual']) > (part2['grid']-part2['feed'])]
part2p['adjusted_cost'] = part2p['cost']-(part2p['grid']*part2p['price']*0.25 + (part2p['load_forecast']-part2p['load_actual']-part2p['grid'])*part2p['price']*0.9*0.25)

part2pp = part2[(part2['load_forecast']-part2['load_actual']) <= (part2['grid']-part2['feed'])]
part2pp['adjusted_cost'] = part2pp['cost']-((part2pp['load_forecast']-part2pp['load_actual'])*part2pp['price']*0.25)

df = pd.concat([part1,part2p,part2pp]).sort_index()

df.insert(8, 'battery', df['battery-out']-df['battery-in'])
df.insert(6, 'grid_forecast', df['grid']-df['feed'])
df.insert(6, 'grid_actual', df['grid_forecast']+df['load_actual']-df['load_forecast'])
df.drop(['battery-out','battery-in','grid','feed'], axis=1, inplace=True )
df.to_csv('../result/their_adjusted/g_mip_battery.csv')

#############

df = pd.read_csv('../result/their_adjusted/g_mip_battery.csv', index_col=0)
costlist = list(map(float, pd.read_csv('../result/their_forecast/g_mip_battery_cost.csv')['daily_cost']))
daily_cost = []
for x in range(360):
    daily_cost.append(df.iloc[96*x : 96*(x+1)]['adjusted_cost'].sum())
print(sum(costlist))
print(sum(daily_cost))
pd.DataFrame(data={"daily_cost": costlist, 'adjusted_daily_cost': daily_cost}, columns=["daily_cost",'adjusted_daily_cost']).reset_index(drop=True).to_csv('../result/their_adjusted/g_mip_battery_cost.csv')

##############
