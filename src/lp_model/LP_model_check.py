from gurobipy import *
import pandas as pd



## PARAMETERS
# Note: Some of these parameters are hypotetical and will be changed with true / reasonable values. 

# Initialize Constant Parameters

# Price = pd.read_csv("../../input/price.csv", converters={"price":
#     float})["price"].tolist() # Electriciy prices in TOU Tariff per MegaWattHours in Euros (€)

Price = [7200, 7200, 7200] # Electriciy prices in TOU Tariff per MegaWattHours in Euros (€)
T, dT, n1, n2, SoC_opt, p_back, mis_pen = [len(Price), 0.25, 0.9, 0.9, 0.4, 0.9, 0] # Number of periods in a day, period length in hours (h),
    # efficiencies of -/- converter and -/~ inverter as ratios (#), optimum SoC as ratio (#),
    # back sell price multiplier (#), battery SoC mismatch penalty price (€)
Mg, Mgl, Mgb, Mpvl, Mpvb, Mpvg, Mbl, Mbg = [GRB.INFINITY, GRB.INFINITY, GRB.INFINITY, GRB.INFINITY,
    GRB.INFINITY, GRB.INFINITY, GRB.INFINITY, GRB.INFINITY] # Maximum power flow limits for all lines in MegaWatts (MW)
MCharge = 12 # Maximum Battery stored energy in MegaWattHours (MWh)
ICharge = MCharge # Initial Battery stored energy in MegaWattHours (MWh)



## LP MODEL

# LP Minimization Model Function
def LP_Minimize_Cost(i=0):
    
    global ICharge

    # Read Iterable Parameters
    # Ppv = pd.read_csv("../../input/pvgeneration.csv", converters={"actual": float},
    #     skiprows=range(1,26401+T*i), nrows=T)["actual"].tolist() # Daily PV generation forecast in MegaWatts (MW)
    # Pl = pd.read_csv("../../input/load.csv", converters={"actual": float},
    #     skiprows=range(1,70081+T*i), nrows=T)["actual"].tolist() # Daily Load forecast in MegaWatts (MW)
    Ppv = [10, 10, 10]
    Pl = [10, 10, 10]
    
    m = Model() # Model Object: m

    Pg, Pgl, Pgb, Ppvl, Ppvb, Ppvg, Pbl, Pbg = [[0 for t in range(T)], [0 for t in range(T)], [0 for t in range(T)], [0 for t in range(T)],
        [0 for t in range(T)], [0 for t in range(T)], [0 for t in range(T)], [0 for t in range(T)]] # Decision variables: Power flows in MW
    Charge = [0 for t in range(T)] # Decision variable: Battery stored energy in MWh
    Mismatch, u, v = [[0 for t in range(T)], [0 for t in range(T)], [0 for t in range(T)]]

    # Add Decision Variables
    for t in range(T):
        Pg[t] = m.addVar(lb=0, ub=Mg, vtype=GRB.CONTINUOUS) # Grid input power flow variables
        Pgl[t] = m.addVar(lb=0, ub=Mgl, vtype=GRB.CONTINUOUS) # Grid to Load power flow variables
        Pgb[t] = m.addVar(lb=0, ub=Mgb, vtype=GRB.CONTINUOUS) # Grid to Battery power flow variables
        Ppvl[t] = m.addVar(lb=0, ub=Mpvl, vtype=GRB.CONTINUOUS) # PV to Load power flow variables
        Ppvb[t] = m.addVar(lb=0, ub=Mpvb, vtype=GRB.CONTINUOUS) # PV to Battery power flow variables
        Ppvg[t] = m.addVar(lb=0, ub=Mpvg, vtype=GRB.CONTINUOUS) # PV to Grid (back sell) power flow variables
        Pbl[t] = m.addVar(lb=0, ub=Mbl, vtype=GRB.CONTINUOUS) # Battey to Load power flow variables
        Pbg[t] = m.addVar(lb=0, ub=Mbg, vtype=GRB.CONTINUOUS) # Battey to Grid (back sell) power flow variables
        Charge[t] = m.addVar(lb=0, ub=MCharge, vtype=GRB.CONTINUOUS) # Battety stored energy variables
        Mismatch[t] = m.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS) # Batttery SoC mismatch absolute ratio
        u[t] = m.addVar(lb=-1, ub=1, vtype=GRB.CONTINUOUS) # Batttery SoC mismatch signed ratio
        v[t] = m.addVar(lb=0, ub=max(Mgb,Mbg), vtype=GRB.CONTINUOUS) # Sum of Grid to Battery and Battey to Grid power flows
    
    # Add Linear Constraints
    for t in range(T):
        m.addConstr(Ppvl[t] + Ppvb[t] + Ppvg[t] == Ppv[t]) # KCL constraints for PV
        m.addConstr(Pgl[t] + Pgb[t] == Pg[t]) # KCL constraints for Grid
        m.addConstr(Pgl[t] + n1 * n2 * Ppvl[t] + n2 * Pbl[t] == Pl[t]) # KCL constraints for Load
        if t == 0:
            m.addConstr(ICharge + dT * (n2 * Pgb[t] + n1 * Ppvb[t] - Pbl[t] - Pbg[t]) == Charge[t]) # The first KCL constraint for Battery
        else:
            m.addConstr(Charge[t-1] + dT * (n2 * Pgb[t] + n1 * Ppvb[t] - Pbl[t] - Pbg[t]) == Charge[t]) # Rest of the KCL constraints for Battery
        m.addConstr(v[t] == Pgb[t] + Pbg[t])
        m.addConstr(v[t] == max_(Pgb[t], Pbg[t])) # Only one of Grid to Battery and Battey to Grid power flows is allowed in a period constraints
        m.addConstr(u[t] == Charge[t] / MCharge - SoC_opt)
        m.addConstr(Mismatch[t] == abs_(u[t])) # Mismatch as absolute value of the difference between Batttery SoC and Optimal SoC constraints

    # Optimize Model
    # Set Objective Function
    m.setObjective(sum(((dT * Price[t] * Pg[t]) + (Mismatch[t] * mis_pen) -
        (dT * Price[t] * p_back * (n1 * n2 * Ppvg[t] + n2 * Pbg[t]))) for t in range(T)), GRB.MINIMIZE)
    m.update() # Update Model
    m.optimize() # Run Model Optimization
    
    # Print Results
    if m.status == GRB.Status.OPTIMAL:
        Pg_results = m.getAttr('x', Pg)
        Pgl_results = m.getAttr('x', Pgl)
        Pgb_results = m.getAttr('x', Pgb)
        Ppvl_results = m.getAttr('x', Ppvl)
        Ppvb_results = m.getAttr('x', Ppvb)
        Ppvg_results = m.getAttr('x', Ppvg)
        Pbl_results = m.getAttr('x', Pbl)
        Pbg_results = m.getAttr('x', Pbg)
        Charge_results = m.getAttr('x', Charge)
        Mismatch_results = m.getAttr('x', Mismatch)
        
        ICharge = Charge_results[T-1]
        SoC = [(round(100*Charge_results[t]/MCharge)) for t in range(T)]
        
        print("Optimal solution is found in ", round(m.Runtime,3), " seconds.")
        print("Total_Cost (€) = ", m.objVal)
        # print("ICharge = ", ICharge)
        print("SoC (%): " + " ".join(map(str, SoC)))
        print("Pl: ", Pl)
        print("Ppv: ", Ppv)
        print("Pg: ", Pg_results)
        print("Pgl: ", Pgl_results)
        print("Pgb: ", Pgb_results)
        print("Ppvl: ", Ppvl_results)
        print("Ppvb: ", Ppvb_results)
        print("Ppvg: ", Ppvg_results)
        print("Pbl: ", Pbl_results)
        print("Pbg: ", Pbg_results)
        print("Charge: ", Charge_results)
        print("Mismatch: ", Mismatch_results)
    else:
        print("There is no feasible solution!")
        ICharge = 0
    
    return



## OPTIMIZATION

LP_Minimize_Cost()
input()