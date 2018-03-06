from gurobipy import *
import pandas as pd
import numpy as np
import timeit

Date = pd.read_csv("../../input/load2017.csv")["date"].tolist()
From = pd.read_csv("../../input/load2017.csv")["from"].tolist()
To = pd.read_csv("../../input/load2017.csv")["to"].tolist()
# write = pd.DataFrame(data={"date": Date, "from": From, "to": To})
# write_val = pd.DataFrame(data={"x": 01.01.2018})
# with open("../../input/deneme.csv", "a") as f:
#     write.to_csv(f, header=False, index=False)
#     write_val.to_csv(f["date"], header=False, index=False)

p_tariff = pd.read_csv("../../input/price.csv", converters={"price": float})["price"].tolist()
PV_list = pd.read_csv("../../input/pvgeneration2017.csv", converters={"actual": float})["actual"].tolist()
L_list = pd.read_csv("../../input/load2017.csv", converters={"actual": float})["actual"].tolist()

N, m_back, T, eff, SoC_opt, c_pen, E_max = [len(p_tariff), 0.9, 0.25, 0.9, 0.5, 0.5, 12]
E_init = SoC_opt * E_max
Total_cost = 0 

G_pos_list, G_neg_list, B_pos_list, B_neg_list, E_list, s_list, Cost_list, Daily_cost_list = [
    [], [], [], [], [], [], [], []]

G_pos, G_neg, B_pos, B_neg, E, s, b = [[0 for n in range(N)], [0 for n in range(N)], [0 for n in range(N)], 
        [0 for n in range(N)], [0 for n in range(N)], [0 for n in range(N)], [0 for n in range(N)]]

def Optimize(p, PV, L, strategy=True, j=0):
    
    global G_pos, G_neg, B_pos, B_neg, E, s, b, E_init
    
    m = Model()

    for n in range(N):
        G_pos[n] = m.addVar(lb=0, vtype=GRB.CONTINUOUS)
        G_neg[n] = m.addVar(lb=0, vtype=GRB.CONTINUOUS)
        B_pos[n] = m.addVar(lb=0, vtype=GRB.CONTINUOUS)
        B_neg[n] = m.addVar(lb=0, vtype=GRB.CONTINUOUS)
        E[n] = m.addVar(lb=0, vtype=GRB.CONTINUOUS)
        s[n] = m.addVar(lb=0, vtype=GRB.CONTINUOUS)
        b[n] = m.addVar(lb=0, vtype=GRB.BINARY)

    for n in range(N):
        m.addConstr(G_pos[n] - G_neg[n] + eff * (PV[n] + B_pos[n] - B_neg[n]) - L[n] == 0)
        if n==0:
            m.addConstr(E_init - (B_pos[n] - B_neg[n]) * T - E[n] == 0)
            m.addConstr(G_neg[n] * T <= (eff * E_init + eff * PV[n] * T) * (1-b[n]))
        else:
            m.addConstr(E[n-1] - (B_pos[n] - B_neg[n]) * T - E[n] == 0)
            m.addConstr(G_neg[n] * T <= (eff * E[n-1] + eff * PV[n] * T) * (1-b[n]))
        m.addConstr(eff * G_pos[n] * T <= (E_max + eff * L[n] * T) * b[n])
        m.addConstr(E[n] <= E_max)
        m.addConstr(E[n] - SoC_opt * E_max <= s[n])
        m.addConstr(SoC_opt * E_max - E[n] <= s[n])

    m.setObjective((sum(p[n] * (G_pos[n] - m_back * G_neg[n]) * T + c_pen * s[n] for n in range(N))
        - p[N-1] * (E[N-1] - E_init)), GRB.MINIMIZE)
    
    m.update()
    m.setParam("OutputFlag", False)
    m.optimize()

    result = 0
    if m.status == GRB.status.OPTIMAL:
        Result(m, p, strategy, j)
        if strategy:
            result = m.objVal
            E_init = E[N-1].x
        else:
            result = (p[0] * (G_pos[0].x - m_back * G_neg[0].x) * T) + (c_pen * s[0].x) - (p[j] * E[0].x - p[j-1] * E_init)
            E_init = E[0].x
        return result
    else:
        print("infeasible")
        return result



def Result(m, p, strategy=True, j=0):
    
    global G_pos_list, G_neg_list, B_pos_list, B_neg_list, E_list, s_list, Cost_list

    if strategy:
        G_pos_list += np.ndarray.tolist(np.around(np.asarray(m.getAttr('x', G_pos)), decimals=1))
        G_neg_list += m.getAttr('x', G_neg)
        B_pos_list += m.getAttr('x', B_pos)
        B_neg_list += m.getAttr('x', B_neg)
        E_list += m.getAttr('x', E)
        s_list += m.getAttr('x', s)
        Cost_list += [(p[0] * (G_pos[0].x - m_back * G_neg[0].x) * T) + (c_pen * s[0].x)
            - (p[0] * E[0].x - p[N-1] * E_init)] + [(p[n] * (G_pos[n].x - m_back * G_neg[n].x) * T)
            + (c_pen * s[n].x) - (p[n] * E[n].x - p[n-1] * E[n-1].x) for n in range(1,N)]
    else:
        G_pos_list.append(round(G_pos[0].x,1))
        G_neg_list.append(G_neg[0].x)
        B_pos_list.append(B_pos[0].x)
        B_neg_list.append(B_neg[0].x)
        E_list.append(E[0].x)
        s_list.append(s[0].x)
        Cost_list.append((p[0] * (G_pos[0].x - m_back * G_neg[0].x) * T)
            + (c_pen * s[0].x) - (p[j] * E[0].x - p[j-1] * E_init))



def Prompt(m):
    print("**********R*E*S*U*L*T*S**********")
    print("initial = ", E_init)
    print("charge = ", m.getAttr('x', B_neg))
    print("discharge = ", m.getAttr('x', B_pos))
    print("energy = ", m.getAttr('x', E))
    print("missmatch = ", m.getAttr('x', s))
    print("bought = ", m.getAttr('x', G_pos))
    print("sold = ", m.getAttr('x', G_neg))
    print("cost = ", m.objVal)


# 
def Write():
    write_all = pd.DataFrame(data={"date": Date[:192], "from": From[:192], "to": To[:192], "load": L_list[:192], "pv": PV_list[:192], 
        "grid": G_pos_list, "feed": G_neg_list, "battery-in": B_neg_list, "battery-out": B_pos_list,
        "energy": E_list, "mismatch": s_list, "cost": Cost_list}, columns=["date","from","to",
        "load", "pv", "grid", "feed", "battery-in", "battery-out", "energy", "mismatch", "cost"])
    write_all.to_csv("../../input/deneme_all.csv", index=None)
    # write_cost = pd.DataFrame(data={"cost": Daily_cost_list})
    # with open("../../input/deneme_cost.csv", "a") as file:
    #     write_cost.to_csv(file)



def Simulate(strategy=True, ndays=1):
    if strategy:
        p = p_tariff
        for i in range(ndays):
            PV = PV_list[N*i:N*i+N]
            L = L_list[N*i:N*i+N]
            Daily_cost_list.append(Optimize(p, PV, L, strategy))
        Total_cost = sum(Daily_cost_list)
        print("LP daily cost list: ", Daily_cost_list)
        print("LP total cost: ", Total_cost)
    else:
        p_seq = p_tariff + p_tariff
        for i in range(ndays):
            PV_seq = PV_list[N*i:N*(i+2)]
            L_seq = L_list[N*i:N*(i+2)]
            Daily_cost = 0
            for j in range(N):
                PV = PV_seq[j:j+N]
                L = L_seq[j:j+N]
                p = p_seq[j:j+N]
                Daily_cost += Optimize(p, PV, L, strategy, j)
            Daily_cost_list.append(Daily_cost)
        Total_cost = sum(Daily_cost_list)
        print("DP daily cost list: ", Daily_cost_list)
        print("DP total cost: ", Total_cost)
    # Write()



# Simulate(True,365)
# Simulate(False,364)


start = timeit.default_timer()
Simulate()
end = timeit.default_timer()
print(G_pos_list)
print(end-start)
