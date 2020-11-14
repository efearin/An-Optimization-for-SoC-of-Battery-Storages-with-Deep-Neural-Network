from gurobipy import *
import pandas as pd
import numpy as np



# SET SCENATIO AND COUNTRY
"""
Strategy: True for LP, False for DP
Scenario: True for battery with PV, False for battery only
Country: True for Turkey price tariff, False for Germany price tariff
"""
Strategy, Scenario, Country = [True, True, True]



# x = np.array(pd.read_csv("../result/t_mip_battery_pv_cost.csv", header=0, index_col=0)).reshape(360,2)
# print(x.sum(axis=0))

# DATA READING AND MANIPULATION
if Country:
    p_tariff = pd.read_csv("../data/price.csv", header=0)["turkey"].tolist()
else:
    p_tariff = pd.read_csv("../data/price.csv", header=0)["germany"].tolist()

Date = pd.read_csv("../data/actual.csv", header=0)["date"].tolist()
From = pd.read_csv("../data/actual.csv", header=0)["from"].tolist()
To = pd.read_csv("../data/actual.csv", header=0)["to"].tolist()
Act_L = np.array(pd.read_csv("../data/actual.csv", header=0)["load"].values) * 0.00001885071
Act_PV = np.array(pd.read_csv("../data/actual.csv", header=0)["pv"].values) * 0.00036682367
L_actual_list = Act_L.tolist()
PV_actual_list = Act_PV.tolist()

L_df = pd.read_csv("../data/load.csv", header=0, index_col=0) #34560
L_df = L_df[L_df.columns.values[-96:]]
L_ar = np.array(L_df.values)
L_ar = L_ar[96:34656] * 0.00001885071

PV_df = pd.read_csv("../data/pv.csv", header=0, index_col=0) #34560
PV_df = PV_df[PV_df.columns.values[-96:]]
PV_ar = np.array(PV_df.values)
if Scenario:
    PV_ar = PV_ar[96:34656] * 0.00036682367 # Battery with PV (PV included)
else:
    PV_ar = PV_ar[96:34656] * 0 # Battery only (PV not included)



# MODEL STRUCTURE

# Model parameters
"""
N: Number of periods in a day
m_back: Price multipler for sold energy
T: Length of a period (h)
eff: Inverter efficiency rate
I_max: Inverter capacity limit for the battery (kW)
B_max: Maximum charge/discharge power limit for the battery (kW)
E_max: Maximum energy storage limit for the battery (kWh)
E_init: Initial stored energy in the battery (kWh)
"""
N, m_back, T, eff, I_max, B_max, E_max, E_init = [96, 0.9, 0.25, 0.9, 3, 3, 6, 0]

# Model decision variables
G_pos, G_neg, B_pos, B_neg, E, b1, b2, b3 = [[0 for _ in range(N)], [0 for _ in range(N)], [0 for _ in range(N)], 
    [0 for _ in range(N)], [0 for _ in range(N)], [False for _ in range(N)], [False for _ in range(N)], [False for _ in range(N)]]

# Placeholder lists for values of model decision variables and calculated periodical and daily costs for both LP and DP
G_pos_list_LP, G_neg_list_LP, B_pos_list_LP, B_neg_list_LP, E_list_LP, Cost_list_LP, Daily_cost_list_LP = [[],[],[],[],[],[],[]]
G_pos_list_DP, G_neg_list_DP, B_pos_list_DP, B_neg_list_DP, E_list_DP, Cost_list_DP, Daily_cost_list_DP = [[],[],[],[],[],[],[]]
L_forecast_list_LP, PV_forecast_list_LP, L_forecast_list_DP, PV_forecast_list_DP, Cost_list_blind, Daily_cost_list_blind = [[],[],[],[],[],[]]
Total_cost_LP, Total_cost_DP, Total_cost_blind = [0, 0, 0]



# OPTIMIZATION FUNCTION
"""
    p: Price Tariff (List)
    PV: PV (List)
    L: Load (List)
    strategy: True for LP, False for DP
"""
def Optimize(p, PV, L, strategy=Strategy):
    global G_pos, G_neg, B_pos, B_neg, E, b1, b2, b3, E_init # To enable updating model decision variables and initial stored energy
    
    # Create model
    m = Model()

    # Add decision variables to the model
    for n in range(N):
        G_pos[n] = m.addVar(lb=0, vtype=GRB.CONTINUOUS) # Decision variables for power flow (kW) drawn from the grid (buy)
        G_neg[n] = m.addVar(lb=0, ub=I_max, vtype=GRB.CONTINUOUS) # Decision variables for power flow (kW) fed into the grid (sell)
        B_pos[n] = m.addVar(lb=0, ub=B_max, vtype=GRB.CONTINUOUS) # Decision variables for power flow (kW) from the battery (discharge)
        B_neg[n] = m.addVar(lb=0, ub=B_max, vtype=GRB.CONTINUOUS) # Decision variables for power flow (kW) into the battery (charge)
        E[n] = m.addVar(lb=0, ub=E_max, vtype=GRB.CONTINUOUS) # Decision variables for instantaneous stored energy (kWh) in the battery (storage)
        b1[n] = m.addVar(vtype=GRB.BINARY) # Pseudo decision variables to prevent buying from and selling into the grid simultaneosly
        b2[n] = m.addVar(vtype=GRB.BINARY) # Pseudo decision variables to prevent charging and discharging the battery simultaneosly
        b3[n] = m.addVar(vtype=GRB.BINARY) # Pseudo decision variables to ensure reciprocity of the inverter efficiency and to satisfy KCL
    
    # Add constraints to the model
    for n in range(N):
        if n==0:
            m.addConstr(E_init - (B_pos[n] - B_neg[n]) * T - E[n] == 0) # Energy flow balance (KCL) for the battety constraints (intitial case) (C1.a)
        else:
            m.addConstr(E[n-1] - (B_pos[n] - B_neg[n]) * T - E[n] == 0) # Energy flow balance (KCL) for the battety constraints (C1.b)
        m.addConstr(G_neg[n] <= I_max * (1-b1[n])) # Simultaneous buying and selling avoidance constraints (C2.1)
        m.addConstr(eff * G_pos[n] <= (I_max + eff * L[n]) * b1[n]) # Simultaneous buying and selling avoidance constraints (C2.2)
        m.addConstr(B_pos[n] <= B_max * b2[n]) # Simultaneous charging and discharging avoidance constraints (C3.1)
        m.addConstr(B_neg[n] <= B_max * (1-b2[n])) # Simultaneous charging and discharging avoidance constraints (C3.2)
        m.addConstr(PV[n] + B_pos[n] - B_neg[n] <= I_max * b3[n]) # Inverter efficiency reciprocity constraints (C4.1)
        m.addConstr(I_max * (b3[n]-1) <= PV[n] + B_pos[n] - B_neg[n]) # Inverter efficiency reciprocity constraints (C4.2)
        m.addConstr(G_pos[n] - G_neg[n] - L[n] + eff * (PV[n] + B_pos[n] - B_neg[n]) * b3[n] + (1/eff) * (PV[n] + B_pos[n] - B_neg[n]) * (1-b3[n]) == 0)
            # Inverter efficiency reciprocity constraints (C4.3) and power flow balance (KCL) for the overall nanogrid system constraints (C5)
    
    # Set objective function of the model
    m.setObjective(quicksum(p[n] * (G_pos[n] - m_back * G_neg[n]) * T for n in range(N)), GRB.MINIMIZE) 
    
    m.update() # Update the model with changes
    m.setParam("OutputFlag", False) # Disable output info
    m.optimize() # Optimize the model

    # Get optimization results
    result = 0
    if m.status == GRB.status.OPTIMAL:
        Result(m, p, strategy)
        if strategy:
            result = m.objVal
            E_init = E[N-1].x
        else:
            result = p[0] * (G_pos[0].x - m_back * G_neg[0].x) * T
            E_init = E[0].x
        return result
    else:
        print("infeasible")
        return result



# SIMULATION FUNCTION
"""
    strategy: True for LP, False for DP
    ndays: Range of simulation (days)
"""
def Simulate(strategy=Strategy, ndays=1):
    global Daily_cost_list_LP, Daily_cost_list_DP, Total_cost_LP, Total_cost_DP, E_init
    global L_forecast_list_LP, PV_forecast_list_LP, L_forecast_list_DP, PV_forecast_list_DP
    if strategy:
        L_forecast_list_LP, PV_forecast_list_LP = [[],[]]
        E_init = 0
        p = p_tariff
        for i in range(ndays):
            L = L_ar[N*i].tolist() # Read daily load forecast
            PV = PV_ar[N*i].tolist() # Read daily PV forecast
            L_forecast_list_LP += L # Append daily load forecast to load forecast list
            PV_forecast_list_LP += PV # Append daily PV forecast to PV forecast list
            Daily_cost_list_LP.append(Optimize(p, PV, L, strategy)) # Append daily cost to daily cost list
        Total_cost_LP = sum(Daily_cost_list_LP) # Calculate total cost
        print("LP daily cost list: ", Daily_cost_list_LP)
        print("LP total cost: ", Total_cost_LP)
    else:
        L_forecast_list_DP, PV_forecast_list_DP = [[],[]]
        E_init = 0
        p_seq = p_tariff + p_tariff # Concatenate price tariff for circular shift
        for i in range(ndays):
            Daily_cost_DP = 0 
            for j in range(N):
                p = p_seq[j:j+N] # Shift price tariff list
                L = L_ar[N*i + j].tolist() # Read instantaneous load forecast
                PV = PV_ar[N*i + j].tolist() # Read instantaneous PV forecast
                L_forecast_list_DP.append(L[0]) # Append instantaneous load forecast to load forecast list
                PV_forecast_list_DP.append(PV[0]) # Append instantaneous PV forecast to PV forecast list
                Daily_cost_DP += Optimize(p, PV, L, strategy) # Calculate daily cost
            Daily_cost_list_DP.append(Daily_cost_DP) # Append daily cost to daily cost list
        Total_cost_DP = sum(Daily_cost_list_DP) # Calculate total cost
        print("DP daily cost list: ", Daily_cost_list_DP)
        print("DP total cost: ", Total_cost_DP)
    Write(strategy, ndays)
    return



# RESULT FUNCTION
"""
    m: Model
    p: Price (List)
    strategy: True for LP, False for DP (default: True)
"""
def Result(m, p, strategy=Strategy):
    global G_pos_list_LP, G_neg_list_LP, B_pos_list_LP, B_neg_list_LP, E_list_LP, Cost_list_LP
    global G_pos_list_DP, G_neg_list_DP, B_pos_list_DP, B_neg_list_DP, E_list_DP, Cost_list_DP
    if strategy:
        G_pos_list_LP += m.getAttr('x', G_pos)
        G_neg_list_LP += m.getAttr('x', G_neg)
        B_pos_list_LP += m.getAttr('x', B_pos)
        B_neg_list_LP += m.getAttr('x', B_neg)
        E_list_LP += m.getAttr('x', E)
        Cost_list_LP += [p[n] * (G_pos[n].x - m_back * G_neg[n].x) * T for n in range(N)]
    else:
        G_pos_list_DP += [G_pos[0].x]
        G_neg_list_DP += [G_neg[0].x]
        B_pos_list_DP += [B_pos[0].x]
        B_neg_list_DP += [B_neg[0].x]
        E_list_DP += [E[0].x]
        Cost_list_DP += [p[0] * (G_pos[0].x - m_back * G_neg[0].x) * T]  
    return



# RESULT PROMPTING FUNCTION
"""
    m: Model 
"""
def Prompt(m):
    print("\nRESULT")
    print("initial = ", E_init)
    print("charge = ", m.getAttr('x', B_neg))
    print("discharge = ", m.getAttr('x', B_pos))
    print("energy = ", m.getAttr('x', E))
    print("bought = ", m.getAttr('x', G_pos))
    print("sold = ", m.getAttr('x', G_neg))
    print("cost = ", m.objVal)
    return



# RESULT WRITING FUNCTION
"""
    strategy: True for LP, False for DP
    ndays: Range of simulation (days)
"""
def Write(strategy=Strategy, ndays=1):
    if strategy:
        if Scenario:
            wlp = pd.DataFrame(data={"date": Date[:N*ndays], "from": From[:N*ndays], "to": To[:N*ndays], "load_actual": L_actual_list[:N*ndays],
                "load_forecast": L_forecast_list_LP, "pv_actual": PV_actual_list[:N*ndays], "pv_forecast": PV_forecast_list_LP,
                "grid": G_pos_list_LP, "feed": G_neg_list_LP, "battery-in": B_neg_list_LP, "battery-out": B_pos_list_LP,
                "energy": E_list_LP, "cost": Cost_list_LP}, columns=["date","from","to", "load_actual", "load_forecast",
                "pv_actual", "pv_forecast", "grid", "feed", "battery-in", "battery-out", "energy", "cost"])
            if Country:
                wlp.to_csv("../result/forecast/t_mip_battery_pv.csv")
            else:
                wlp.to_csv("../result/forecast/g_mip_battery_pv.csv")
        else:
            wlp = pd.DataFrame(data={"date": Date[:N*ndays], "from": From[:N*ndays], "to": To[:N*ndays], "load_actual": L_actual_list[:N*ndays],
                "load_forecast": L_forecast_list_LP, "grid": G_pos_list_LP, "feed": G_neg_list_LP, "battery-in": B_neg_list_LP,
                "battery-out": B_pos_list_LP, "energy": E_list_LP, "cost": Cost_list_LP}, columns=["date","from","to", "load_actual",
                "load_forecast", "grid", "feed", "battery-in", "battery-out", "energy", "cost"])
            if Country:
                wlp.to_csv("../result/forecast/t_mip_battery.csv")
            else:
                wlp.to_csv("../result/forecast/g_mip_battery.csv")
        wlp_cost = pd.DataFrame(data={"daily_cost": Daily_cost_list_LP}, columns=["daily_cost"])
        if Scenario:
            if Country:
                wlp_cost.to_csv("../result/forecast/t_mip_battery_pv_cost.csv")
            else:
                wlp_cost.to_csv("../result/forecast/g_mip_battery_pv_cost.csv")
        else:
            if Country:
                wlp_cost.to_csv("../result/forecast/t_mip_battery_cost.csv")
            else:
                wlp_cost.to_csv("../result/forecast/g_mip_battery_cost.csv")
    else:
        if Scenario:
            wdp = pd.DataFrame(data={"date": Date[:N*ndays], "from": From[:N*ndays], "to": To[:N*ndays], "load_actual": L_actual_list[:N*ndays],
                "load_forecast": L_forecast_list_DP, "pv_actual": PV_actual_list[:N*ndays], "pv_forecast": PV_forecast_list_DP,
                "grid": G_pos_list_DP, "feed": G_neg_list_DP, "battery-in": B_neg_list_DP, "battery-out": B_pos_list_DP,
                "energy": E_list_DP, "cost": Cost_list_DP}, columns=["date","from","to", "load_actual", "load_forecast",
                "pv_actual", "pv_forecast", "grid", "feed", "battery-in", "battery-out", "energy", "cost"])
            if Country:
                wdp.to_csv("../result/forecast/t_dynamic_mip_battery_pv.csv")
            else:
                wdp.to_csv("../result/forecast/g_dynamic_mip_battery_pv.csv")
        else:
            wdp = pd.DataFrame(data={"date": Date[:N*ndays], "from": From[:N*ndays], "to": To[:N*ndays], "load_actual": L_actual_list[:N*ndays],
                "load_forecast": L_forecast_list_DP, "grid": G_pos_list_DP, "feed": G_neg_list_DP, "battery-in": B_neg_list_DP,
                "battery-out": B_pos_list_DP, "energy": E_list_DP, "cost": Cost_list_DP}, columns=["date","from","to", "load_actual",
                "load_forecast", "grid", "feed", "battery-in", "battery-out", "energy", "cost"])
            if Country:
                wdp.to_csv("../result/forecast/t_dynamic_mip_battery.csv")
            else:
                wdp.to_csv("../result/forecast/g_dynamic_mip_battery.csv")
        wdp_cost = pd.DataFrame(data={"daily_cost": Daily_cost_list_DP}, columns=["daily_cost"])
        if Scenario:
            if Country:
                wdp_cost.to_csv("../result/forecast/t_dynamic_mip_battery_pv_cost.csv")
            else:
                wdp_cost.to_csv("../result/forecast/g_dynamic_mip_battery_pv_cost.csv")
        else:
            if Country:
                wdp_cost.to_csv("../result/forecast/t_dynamic_mip_battery_cost.csv")
            else:
                wdp_cost.to_csv("../result/forecast/g_dynamic_mip_battery_cost.csv")
    return



# MAIN
Simulate(Strategy,360)