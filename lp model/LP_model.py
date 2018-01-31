from gurobipy import *
import pandas as pd

# PARAMETERS

# Initialize Constant Parameters
# Note: Some of these parameters are hypotetical and will be changed with true / reasonable values. 
dT = 0.25 # Period length in Hours (h)
T = 96 # Number of periods
n1, n2 = [0.9, 0.9] # Efficiencies of -/- converter and -/~ inverter as Ratio (#)
Mgl, Mgb, Mpvl, Mpvb, Mbl = [100000, 100000, 100000, 100000, 100000] # Maximum power flow limits for all lines in MegaWatts (MW)
Msoc = 500000 # Maximum SoC in MegaWattHours (MWh)
SoC0 = 0 # Starting SoC in MegaWattHours (MWh)

# Read and Initialize Iterable Parameters
df=pd.read_csv("actual_pvgeneration.csv", converters={"actual_pvgeneration": float}, nrows=96)
Ppv_List = df["actual_pvgeneration"].tolist() # Daily PV generation forecast in MegaWatts (MW)
df=pd.read_csv("actual_load.csv", converters={"actual_load": float}, nrows=96)
Pl_List = df["actual_load"].tolist() # Daily load forecast in MegaWatts (MW)
df=pd.read_csv("costs_mwh.csv", converters={"costs_mwh": float})
C_List = df["costs_mwh"].tolist() # Electriciy prices in TOU Tariff per MegaWattHours in Euros (€)

# LP MODEL

# LP Minimization Model Function
# Note: This function will be called coninuously for each period
def LP_Minimize_Cost(Ppv, Pl, C, Initial_SoC):
    m = Model() # Model Object: m
    
    Pg, Pgl, Pgb, Ppvl, Ppvb, Pbl = [[0 for t in range(T)], [0 for t in range(T)], [0 for t in range(T)],
        [0 for t in range(T)], [0 for t in range(T)], [0 for t in range(T)]] # Decision variables: Power flows in MW
    SoC = [0 for t in range(T)] # Decision variable: State of Charge in MWh

    # Add Decision Variables
    for t in range(T):
        Pg[t] = m.addVar(lb=0, vtype=GRB.CONTINUOUS)
        Pgl[t] = m.addVar(lb=0, ub=Mgl, vtype=GRB.CONTINUOUS)
        Pgb[t] = m.addVar(lb=0, ub=Mgb, vtype=GRB.CONTINUOUS)
        Ppvl[t] = m.addVar(lb=0, ub=Mpvl, vtype=GRB.CONTINUOUS)
        Ppvb[t] = m.addVar(lb=0, ub=Mpvb, vtype=GRB.CONTINUOUS)
        Pbl[t] = m.addVar(lb=0, ub=Mbl, vtype=GRB.CONTINUOUS)
        SoC[t] = m.addVar(lb=0, ub=Msoc, vtype=GRB.CONTINUOUS)

    # Add Linear Constraints
    for t in range(T):
        m.addConstr(Pgl[t] + Pgb[t] == Pg[t])
        m.addConstr(Ppvl[t] + Ppvb[t] == Ppv[t])
        m.addConstr(Pgl[t] + n1 * n2 * Ppvl[t] + n2 * Pbl[t] == Pl[t])
        if t == 0:
            m.addConstr(Initial_SoC + dT * (n2 * Pgb[t] + n1 * Ppvb[t] - Pbl[t]) == SoC[t])
        else:
            m.addConstr(SoC[t-1] + dT * (n2*Pgb[t] + n1*Ppvb[t] - Pbl[t]) == SoC[t])
    
    # Optimize Model
    m.setObjective(sum((dT * Pg[t] * C[t]) for t in range(T)), GRB.MINIMIZE) # Set Objective Function
    m.update() # Update Model
    m.optimize() # Run Model Optimization
    
    # Print Results
    if m.status == GRB.Status.OPTIMAL:
        Pg_results = m.getAttr('x', Pg)
        Pgl_results = m.getAttr('x', Pgl)
        Pgb_results = m.getAttr('x', Pgb)
        Ppvl_results = m.getAttr('x', Ppvl)
        Ppvb_results = m.getAttr('x', Ppvb)
        Pbl_results = m.getAttr('x', Pbl)
        SoC_results = m.getAttr('x', SoC)
        
        s = [(round(100*SoC_results[t]/Msoc)) for t in range(T)]
        print("SoC (%):" + " ".join(map(str, s)))
        print("Total_Cost (€) = ", m.objVal)
    else:
        print("NO FEASIBLE SOLUTION!")    
    


LP_Minimize_Cost(Ppv_List, Pl_List, C_List, SoC0)