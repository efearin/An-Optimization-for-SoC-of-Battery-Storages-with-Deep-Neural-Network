from gurobipy import *

# PARAMETERS

# Constant Parameters
# Note: Some of these parameters are hypotetical and will be changed with true / reasonable values. 
dT = 0.25 # Period length in Hours (h)
T = 96 # Number of periods
n1, n2 = [0.9, 0.9] # Efficiencies of -/- converter and -/~ inverter as Ratio (#)
Mg, Mgl, Mgb, Mpvl, Mpvb, Mbl = [100000, 100000, 100000, 100000, 100000, 100000] # Maximum power flow limits for all lines in MegaWatts (MW)
Msoc = 200000 # Maximum SoC in MegaWattsHours (MWh)
SoC0 = 0 # Starting SoC in MegaWattsHours (MWh)

# Iterable Parameters
# Note: These parameters will be read from file.
Ppv_List = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.35, 6.94, 25.49, 65.63, 116.35, 176.04, 252.12, 335.11, 414.09, 488.87, 532.86, 595.87, 675.18, 689.83, 695.64, 713.65, 681.54, 654.22, 620.4, 598.95, 617.41, 604.52, 613.8, 598.69, 616.03, 617.59, 543.81, 475.65, 368.28, 293.54, 211.38, 135.76, 66.08, 16.35, 0.39, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Pl_List = [16775, 16527, 16185, 16003, 15680, 15560, 15265, 15159, 14925, 14861, 14846, 15007, 15061, 15005, 15137, 15095, 14960, 14944, 15006, 15089, 15117, 15063, 15092, 15257, 15098, 15004, 15064, 15241, 15390, 15424, 15616, 15850, 16299, 16380, 16680, 17003, 17320, 17731, 18190, 18360, 18875, 18993, 19380, 19555, 19930, 20144, 20360, 20600, 20679, 20637, 20519, 20271, 20377, 20212, 19983, 19857, 19863, 19727, 19794, 19627, 19784, 19828, 19734, 19915, 19960, 20083, 20546, 20895, 21133, 21239, 21454, 21479, 21378, 21249, 21240, 20815, 20637, 20402, 20070, 19710, 19378, 18979, 18850, 18303, 18049, 17958, 17829, 17729, 17709, 17665, 17380, 17273, 17025, 16955, 16773, 16853]
C_List = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]



# MODEL

# Model Function
# Note: This function will be called for each period
def Minimum_Cost(Ppv, Pl, C, Initial_SoC):
    m = Model() # Model Object: m
    
    Pg, Pgl, Pgb, Ppvl, Ppvb, Pbl = [[0 for t in range(T)], [0 for t in range(T)], [0 for t in range(T)],
        [0 for t in range(T)], [0 for t in range(T)], [0 for t in range(T)]] # Decision variables: Power flows in MW
    SoC = [0 for t in range(T)] # Decision variable: State of Charge in MWh

    # Add Decision Variables
    for t in range(T):
        Pg[t] = m.addVar(lb=0, ub=Mg, vtype=GRB.CONTINUOUS)
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
        print("SoC(%):" + " ".join(map(str, s)))
        print("Total_Cost = ", m.objVal)
    else:
        print("NO FEASIBLE SOLUTION!")    
    
    return None

Minimum_Cost(Ppv_List, Pl_List, C_List, SoC0)