from gurobipy import *
import pandas as pd



## PARAMETERS
# Note: Some of these parameters are hypotetical and will be changed with true / reasonable values. 

# Initialize Constant Parameters
dT = 0.25 # Period length in Hours (h)
T = 96 # Number of periods in a ay
n1, n2 = [0.9, 0.9] # Efficiencies of -/- converter and -/~ inverter as Ratio (#)
Mg, Mgl, Mgb, Mpvl, Mpvb, Mbl = [GRB.INFINITY, GRB.INFINITY, GRB.INFINITY,
    GRB.INFINITY, GRB.INFINITY, GRB.INFINITY] # Maximum power flow limits for all lines in MegaWatts (MW)
Price = pd.read_csv("../../input/price.csv", converters={"price":
    float})["price"].tolist() # Electriciy prices in TOU Tariff per MegaWattHours in Euros (€)
MCharge = 500000 # Maximum Battery stored energy in MegaWattHours (MWh)
ICharge = MCharge/2 # Initial Battery stored energy in MegaWattHours (MWh)



## LP MODEL

# LP Minimization Model Function
def LP_Minimize_Cost(i=0):
    
    global ICharge

    # Read Iterable Parameters
    Ppv = pd.read_csv("../../input/pvgeneration.csv", converters={"actual": float},
        skiprows=range(1,26401+T*i), nrows=T)["actual"].tolist() # Daily PV generation forecast in MegaWatts (MW)
    Pl = pd.read_csv("../../input/load.csv", converters={"actual": float},
        skiprows=range(1,70081+T*i), nrows=T)["actual"].tolist() # Daily Load forecast in MegaWatts (MW)
    
    m = Model() # Model Object: m

    Pg, Pgl, Pgb, Ppvl, Ppvb, Pbl = [[0 for t in range(T)], [0 for t in range(T)], [0 for t in range(T)],
        [0 for t in range(T)], [0 for t in range(T)], [0 for t in range(T)]] # Decision variables: Power flows in MW
    Charge = [0 for t in range(T)] # Decision variable: Battery stored energy in MWh

    # Add Decision Variables
    for t in range(T):
        Pg[t] = m.addVar(lb=0, ub=Mg, vtype=GRB.CONTINUOUS) # Grid input power flow variables
        Pgl[t] = m.addVar(lb=0, ub=Mgl, vtype=GRB.CONTINUOUS) # Grid to Load power flow variables
        Pgb[t] = m.addVar(lb=0, ub=Mgb, vtype=GRB.CONTINUOUS) # Grid to Battery power flow variables
        Ppvl[t] = m.addVar(lb=0, ub=Mpvl, vtype=GRB.CONTINUOUS) # PV to Load power flow variables
        Ppvb[t] = m.addVar(lb=0, ub=Mpvb, vtype=GRB.CONTINUOUS) # PV to Battery power flow variables
        Pbl[t] = m.addVar(lb=0, ub=Mbl, vtype=GRB.CONTINUOUS) # Battey to Load power flow variables
        Charge[t] = m.addVar(lb=0, ub=MCharge, vtype=GRB.CONTINUOUS) # Battety stored energy variables
    
    # Add Linear Constraints
    for t in range(T):
        m.addConstr(Pgl[t] + Pgb[t] == Pg[t]) # KCL constraints for Grid
        m.addConstr(Ppvl[t] + Ppvb[t] <= Ppv[t]) # KCL constraints for PV
        m.addConstr(Pgl[t] + n1 * n2 * Ppvl[t] + n2 * Pbl[t] == Pl[t]) # KCL constraints for Load
        if t == 0:
            m.addConstr(Charge[t] + dT * (n2 * Pgb[t] + n1 * Ppvb[t] - Pbl[t]) == ICharge) # The first KCL constraint for Battery
        else:
            m.addConstr(Charge[t-1] + dT * (n2*Pgb[t] + n1*Ppvb[t] - Pbl[t]) == Charge[t]) # Rest of the KCL constraints for Battery
    
    # Optimize Model
    m.setObjective(sum((dT * Pg[t] * Price[t]) for t in range(T)), GRB.MINIMIZE) # Set Objective Function
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
        Charge_results = m.getAttr('x', Charge)
        
        ICharge = Charge_results[T-1]
        SoC = [(round(100*Charge_results[t]/MCharge)) for t in range(T)]
        print("ICharge = ", ICharge)
        print("SoC (%):" + " ".join(map(str, SoC)))
        print("Total_Cost (€) = ", m.objVal)
    else:
        print("NO FEASIBLE SOLUTION!")
        ICharge = 0
    
    return



## OPTIMIZATION

# Optimization Loop
for i in range(10):
    LP_Minimize_Cost(i)