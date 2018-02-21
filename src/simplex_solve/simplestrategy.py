import simplex

class Point:

    # dimention 4*len(pv_forecast_list) <1pvl, 1pvb, 1bl, 1gb,
    #                                    2pvl, 2pvb, 2bl, 2gb,
    #                                    3pvl, 3pvb, 3bl, 3gb,
    #                                    4pvl, 4pvb, 4bl, 4gb>

    def __init__(self, coordinate: list, cost=None):
        self.cost = cost
        self.coordinate = coordinate

    def __sub__(self, point):
        summation = 0
        for x in range(0, len(self.coordinate)):
            summation += (self.coordinate[x] - point.coordinate[x]) ** 2
        return summation ** 0.5

    def __str__(self):
        string = 'All in power\n'
        current_SoC=SoC
        for x in range (0,len(pv_forecast_list)):
            pvc = (pv_forecast_list[x] - (self.coordinate[4 * x + 1] + self.coordinate[4 * x]*inverse_converter_efficiency))*converter_efficiency
            gl = load_forecast_list[x] - (self.coordinate[4 * x + 2] + self.coordinate[4 * x])
            # TODO aşağıdaki current SoC updatesi il birleştirilebilir
            if self.coordinate[4*x+3]>0:
                conversion_loss = ((self.coordinate[4*x]+self.coordinate[4*x+2]+pvc)*inverse_converter_efficiency + self.coordinate[4*x+3])*(1-converter_efficiency)
            else:
                conversion_loss = (self.coordinate[4 * x] + self.coordinate[4 * x + 2] + pvc + abs(self.coordinate[4 * x + 3])) *  (inverse_converter_efficiency - 1)
            string += 'SoC: '+str(format(current_SoC,'.2f'))+'\n'
            string += str(x+1)+"pv-l: "+str(format(self.coordinate[4*x]/t,'.2f'))+" "+\
                      str(x+1)+"pv-b: "+str(format(self.coordinate[4*x+1]/t,'.2f'))+" "+\
                      str(x+1)+"b-l: "+str(format(self.coordinate[4*x+2]/t,'.2f'))+" "+\
                      str(x+1)+"g-b: "+str(format(self.coordinate[4*x+3]/t,'.2f'))+" "+\
                      str(x+1)+"pv-c: "+str(format(pvc/t,'.2f'))+" "+\
                      str(x+1)+"g-l: "+str(format(gl/t,'.2f'))+\
                      "    conversion loss: "+str(format(conversion_loss/t,'.2f'))+"\n"
            if self.coordinate[3] >= 0:
                current_SoC = (battery_capacity * current_SoC+ self.coordinate[4*x+1] + self.coordinate[4*x+3] * converter_efficiency - self.coordinate[4*x+2] * inverse_converter_efficiency) / battery_capacity
            else:
                current_SoC = (battery_capacity * current_SoC+ self.coordinate[4*x+1] + (self.coordinate[4*x+3] - self.coordinate[4*x+2]) * inverse_converter_efficiency) / battery_capacity
            # current_SoC = (battery_capacity * current_SoC + self.coordinate[4*x+3] + self.coordinate[4*x+1] - self.coordinate[4*x+2]) / battery_capacity
        string += 'SoC: ' + str(format(current_SoC, '.2f')) + '\n'
        string += "cost: " + str(self.cost)
        return string

    def get_point_symmetric_to_points_with_scale(self, point1, point2, scale):
        mid_point = point1.get_mid_to_point(point2)
        coordinate = []
        for x in range(0, len(self.coordinate)):
            if self.coordinate[x] <= mid_point.coordinate[x]:
                coordinate.append(self.coordinate[x] + (self.coordinate[x] - mid_point.coordinate[x]) * scale)
            else:
                coordinate.append(self.coordinate[x] - (self.coordinate[x] - mid_point.coordinate[x]) * scale)
        return Point(coordinate)

    def get_mid_to_point(self, point):
        coordinate = []
        for x in range(0, len(self.coordinate)):
            coordinate.append((self.coordinate[x] + point.coordinate[x]) / 2)
        return Point(coordinate)

class Triangle:

    def __init__(self, a: Point, b: Point, c: Point):
        self.best, self.mid, self.worst = sorted([a, b, c], key=lambda x: x.cost, reverse=False)
        self.area = Triangle.get_linear_area(self)

    def update_point_characteristics(self):
        self.best, self.mid, self.worst = sorted([self.best, self.mid, self.worst], key=lambda x: x.cost, reverse=False)

    def get_linear_area(self):
        a = self.best - self.mid
        b = self.mid - self.worst
        c = self.worst - self.best
        # s = (a+b+c)/2
        # return abs(s*(s-a)*(s-b)*(s-c))**0.5
        # return (self.best-self.mid) + (self.mid-self.worst) + (self.worst-self.best)

        return max(a, b, c)

def get_constants():
    global battery_capacity, optimal_SoC, SoC_mismatch_penalty, back_sell_price_multiplier, converter_efficiency, t, \
        battery_capacity, inverse_converter_efficiency, cutoff_cost, cutoff_area
    battery_capacity = 1  # Ah
    battery_voltage = 12  # V
    battery_capacity = battery_capacity * battery_voltage * 3600  # to energy

    optimal_SoC = 0.4  # [0,1]
    SoC_mismatch_penalty = 0  # punishment calculated manually using battery specs TODO must be revalued accordig to energies (multiply with energy to make it comperable with other terms) (100k upper)
    back_sell_price_multiplier = 0.9  # multiplier of the price when power flows from system to grid
    # (back_sell_price_multiplier * price per energy will be earned as cost function decreaser)

    converter_efficiency = 0.9
    inverse_converter_efficiency = 1 / converter_efficiency
    t = 0.25  # h TODO: might be turn into list
    t = t * 3600  # hour to second

    cutoff_cost = 0.00001  # where will it stop to search according to cost
    cutoff_area = 1  # where will it stop according to search triangle area

def validity_check(coordinate: list):
    """
    :param coordinate: length of 4*load_forecast
    :return: point if coordinate is valid in universe, None if not
    """
    pvc_list=[]
    gl_list=[]
    SoC_list=[SoC]
    cost=0

    for l in range (0,len(pv_forecast_list)):

        # branch -not in coordinates- calculation
        pvc_list.append( (pv_forecast_list[l] - (coordinate[4*l]*inverse_converter_efficiency + coordinate[4*l+1])) * converter_efficiency )
        gl_list.append(load_forecast_list[l] - (coordinate[4*l+2] + coordinate[4*l]))

        # power violation check
        if (coordinate[4*l] < 0 or coordinate[4*l+1] < 0 or coordinate[4*l+2] < 0 or pvc_list[l] < 0 or gl_list[l] < 0):
            return None

        # SoC update depending on gb sign
        if coordinate[4*l+3]>=0:
            SoC_list.append((battery_capacity * SoC_list[l] + coordinate[4*l+1] + coordinate[4*l+3]*converter_efficiency - coordinate[4*l+2]*inverse_converter_efficiency) / battery_capacity)
        else:
            SoC_list.append((battery_capacity * SoC_list[l] + coordinate[4 * l + 1] + (coordinate[4 * l + 3] - coordinate[4 * l + 2]) * inverse_converter_efficiency) / battery_capacity)

        # SoC violation check
        if (SoC_list[-1] > 1 or SoC_list[-1] < 0):
            return None

        # cost update depending on gb sign
        if coordinate[4*l+3]>=0:
            cost += (price[l] * (gl_list[-1] + coordinate[4*l+3] - pvc_list[-1]*back_sell_price_multiplier)+ abs(0.4 - SoC_list[-1]) * SoC_mismatch_penalty)
        else:
            cost += (price[l] * (gl_list[-1] - (pvc_list[-1] + abs(coordinate[4*l+3])) * back_sell_price_multiplier) + abs(0.4 - SoC_list[-1]) * SoC_mismatch_penalty)

    return Point(coordinate, cost)

def main(_load_forecast_list, _pv_forecast_list, _price, _SoC):

    get_constants()

    global load_forecast_list, pv_forecast_list, SoC, price

    load_forecast_list = _load_forecast_list  # W
    pv_forecast_list = _pv_forecast_list  # W

    SoC = _SoC  # [0,1]
    price = _price  # grid usage price per watt (TODO convert it per energy)

    load_forecast_list = [i * t for i in load_forecast_list]  # to energy
    pv_forecast_list = [i * t for i in pv_forecast_list]  # to energy

    # gb dimentions are always 0 at simplistic approach
    # dimention 4*len(pv_forecast_list) <1pvl, 1pvb, 1bl, 1gb,
    #                                    2pvl, 2pvb, 2bl, 2gb,
    #                                    3pvl, 3pvb, 3bl, 3gb,
    #                                    4pvl, 4pvb, 4bl, 4gb>

    coordinate =[]

    for x in range (0,len(load_forecast_list)):

        pv = pv_forecast_list[x]
        load = load_forecast_list[x]

        # pv to load
        pvl = min(pv*converter_efficiency, load)
        pv -= pvl*inverse_converter_efficiency
        load -= pvl

        pvl, pvb, bl = 0,0,0

        if pv!=0:
            # pv to battery
            pvb = min (battery_capacity*(1-SoC), pv)
            pv -= pvb
            SoC = (battery_capacity*SoC+pvb)/battery_capacity
                # if pv!=0:
                #     pvc = pv*converter_efficiency
                #     pv = 0
        if load!=0:
            bl= min (load, battery_capacity*SoC*converter_efficiency)
            load -=bl
            SoC = (SoC*battery_capacity-bl*inverse_converter_efficiency)/battery_capacity
                # if load !=0:
                #     gl = load
                #     load = 0

        coordinate += [pvl, pvb, bl, 0]
    point = validity_check(coordinate)
    print('\nPOINT (simple strategy)')
    print(point)
    return point
if __name__ == '__main__':
    main([10, 10, 10], [10, 10, 10], [2,2,2], 1)