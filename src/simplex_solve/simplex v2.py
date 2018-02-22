import random
from operator import attrgetter
import time
from multiprocessing import Process, Queue,Pool



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

    def approach_to_point_with_scale (self, point,scale):
        # self is approaching to given point with given scale
        # if scale is 16 for example distance between 2 points divided by 16
        # and new point is returned 15/16*distance away from self and 1/16*distance close to point
        coordinate = []
        for x in range(0, len(self.coordinate)):

            # calculate distance between points for one dimension
            if self.coordinate[x]*point.coordinate[x]>=0:
                distance = abs(self.coordinate[x]-point.coordinate[x])
            else:
                distance = abs(self.coordinate[x])+abs(point.coordinate[x])

            # calculate new point's related dimension
            if self.coordinate[x]>= point.coordinate[x]:
                coordinate.append(point.coordinate[x]+distance*(1/scale))
            else:
                coordinate.append(point.coordinate[x] - distance * (1 / scale))
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

# dimention 4*len(pv_forecast_list) <1pvl, 1pvb, 1bl, 1gb,
#                                    2pvl, 2pvb, 2bl, 2gb,
#                                    3pvl, 3pvb, 3bl, 3gb,
#                                    4pvl, 4pvb, 4bl, 4gb>
# pvl: photovoltaic to load (>0) | pvb: photovoltaic to battery (>0)| bl: battery to load (>0) | gb: grid to battery (might be + or -)
# all converters stick to DC sides
# so dimensions: pvl: after converter | pvb: no converter on line | bl: after converter
#              | gb: on the grid side value and converter located to battery side (converter position change depending on power flow)
#              | pvc: after converter | gl: no converter on line
# All energy !!

# TODO üçgenlerin belli bir boyutu geçmesini engelleyebiliriz 2 katına çıka çıka gitmesin bi yerden sonra sabit kalsın watt cşnsşnden bir değer belki ayarlanabilir
# TODO universe multiplierda point için olanı evrenin bilmemkaçta biri olarak belirliyoruz belki onu da watt cinsinden düşünebiliriz
# TODO all SoC updates could be reform in similar notations

# cost calculation is in validity check

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

def get_universe_multipliers(point_coordinate):
    coordinate = list(point_coordinate)
    # TODO coordinates i sıfırlıyor neden? kopyasını aldım önce çakallıkla düzelttim ama olmamalı sanki zaten
    """
    :param coordinate: dimentions decided up to point(coordinate) will be used to determine of the boundries of newly added dimention (comming one)
    :return: boundry of next added dimention as coordinate multiplier and how far the other triangle points will be as point multiplier
    """

    #Todo Point multipler might be decided more logically
    dimention_to_point_multiplier = 10000

    last_SoC = SoC
    last_full_cycle = []
    prediction_index = 0
    dimention_end = 0

    while len(coordinate)> 0:
        last_full_cycle = coordinate[:4]
        del coordinate [:4]
        if len(last_full_cycle)==4:
            # SoC update depending on gb sign
            if last_full_cycle[3] >= 0:
                last_SoC = (battery_capacity * last_SoC + last_full_cycle[1] + last_full_cycle[3] * converter_efficiency - last_full_cycle[2] * inverse_converter_efficiency) / battery_capacity
            else:
                last_SoC = (battery_capacity * last_SoC + last_full_cycle[1] + (last_full_cycle[3] - last_full_cycle[2]) * inverse_converter_efficiency) / battery_capacity
            prediction_index += 1

    if len(last_full_cycle)==0:
        # pvl will be predicted (pvl dimention boundry)
        dimention_end = min(pv_forecast_list[prediction_index]*converter_efficiency, load_forecast_list[prediction_index])
    if len(last_full_cycle)==1:
        # pvb will be predicted
        dimention_end = pv_forecast_list[prediction_index]-last_full_cycle[0]*inverse_converter_efficiency
    if len(last_full_cycle)==2:
        # bl will be predicted
        dimention_end = load_forecast_list[prediction_index]-last_full_cycle[0]
    if len(last_full_cycle)==3:
        # gb will be predicted
        energy_in_battery = battery_capacity * last_SoC + last_full_cycle[1] - last_full_cycle[2] * inverse_converter_efficiency
        dimention_begin = -energy_in_battery*converter_efficiency
        dimention_end = (battery_capacity - energy_in_battery)*inverse_converter_efficiency
        return [dimention_begin, dimention_end], min(dimention_end, abs(dimention_begin)) / dimention_to_point_multiplier

    return [0,dimention_end], dimention_end / dimention_to_point_multiplier

def get_random_point(initial_coordinate=None):
    """
    :param initial_coordinate: length of 4*load_forecast or None
    :return: if no input given random Point else point close to given coordinate
    """

    # TODO: validity could be integrated as updated heuristics to get_universe_multiplier


    miss_count = 0
    while True:
        coordinates = []
        for x in range(0, len(load_forecast_list*4)):

            coordinate_multiplier, point_multiplier = get_universe_multipliers(coordinates)

            if x%4 != 3:
                # not grid to battery case so >0 always and coordinate multiplier is in the form of [0, k]
                # so the information is k only
                coordinate_multiplier = coordinate_multiplier[1]
                if initial_coordinate is None:
                    coordinates.append(random.uniform(0, 1) * coordinate_multiplier)
                else:
                    coordinates.append(initial_coordinate[x] + random.uniform(-1, 1) * point_multiplier)

            else:
                # grid to battery case so the line could be >0 or <0 means coordinate multiplier is in the form of [-l,k]
                if initial_coordinate is None:
                    coordinates.append(random.uniform(0, 1) * (abs(coordinate_multiplier[0])+coordinate_multiplier[1]) + coordinate_multiplier[0])
                else:
                    coordinates.append(initial_coordinate[x] + random.uniform(-1, 1) * point_multiplier)

        point = validity_check(coordinates)
        if point is not None:
            # if initial_coordinate is None:
            #     print('FULLY random point found after '+ str(miss_count)+ ' attempt')
            # else:
            #     print('CLOSE random point found after '+ str(miss_count)+ ' attempt')
            return point
        miss_count+=1

def get_random_triangle():

    points = []

    points.append(get_random_point())
    points.append(get_random_point(points[0].coordinate))
    while True:
        last_point = get_random_point(points[0].coordinate)
        if last_point.coordinate != points[1].coordinate:
            points.append(last_point)
            return Triangle(points[0], points[1], points[2])

def update_triangle(triangle):
    """
    :param triangle: triangle to be updated
    :return: updated triangle (either worst point of triangle is mirrored with a scale to the mid-best line
             or mid and worst is shrunk to best by half)
    """
    scalars = [16, 8, 4, 2, 1.5, 1.25, 1]

    # move case
    for x in range(0, len(scalars)):
        new_worst_point = triangle.worst.get_point_symmetric_to_points_with_scale(triangle.mid, triangle.best,
                                                                                  scalars[x])
        new_worst_point = validity_check(new_worst_point.coordinate)
        if new_worst_point is not None:
            if new_worst_point.cost < triangle.worst.cost:
                # print('move: ' + str(scalars[x]) + ' ' + str(Triangle(new_worst_point, triangle.mid, triangle.best).worst.cost))
                return Triangle(new_worst_point, triangle.mid, triangle.best)

    # shrink case
    scalars=[2]
    for x in range(0,len(scalars)):
        new_worst_point = triangle.worst.approach_to_point_with_scale(triangle.best,scalars[x])
        new_mid_point = triangle.mid.approach_to_point_with_scale(triangle.best,scalars[x])
        new_worst_point = validity_check(new_worst_point.coordinate)
        new_mid_point = validity_check(new_mid_point.coordinate)
        if (new_worst_point is not None) and (new_mid_point is not None):
            # print('shrink ' + str(Triangle(triangle.best, new_mid_point, new_worst_point).worst.cost))
            return Triangle(triangle.best, new_mid_point, new_worst_point)

    # triangel stucked
    return None

# def update_point(point):
#     # TODO move it to point class
#     """
#     change path if g-b-l path exists as g-l then if pv-b-l exists as pv-l
#     :param point: Point
#     :return: Point with different coordinates but cost is same
#     """
#
#     for q in range(0,len(pv_forecast_list)):
#
#         pvc = (pv_forecast_list[q] - (point.coordinate[4*q]*inverse_converter_efficiency + point.coordinate[4*q+1])) * converter_efficiency
#         gl = load_forecast_list[q] - (point.coordinate[4 * q + 2] + point.coordinate[4 * q])
#
#         # pv-g-l
#         pvgl = min(pvc, gl)
#         pvc -= pvgl
#         gl -= pvgl
#         point.coordinate[4*q] += pvgl
#
#         if point.coordinate[4*q+3]>=0:
#             pvgbl = min()
#
#         else:
#
#
#
#     return validity_check(point.coordinate)


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
    cutoff_area = 0.00001  # where will it stop according to search triangle area



    #
    # battery_capacity=classes_and_constants.battery_capacity
    # optimal_SoC=classes_and_constants.optimal_SoC
    # SoC_mismatch_penalty=classes_and_constants.SoC_mismatch_penalty
    # back_sell_price_multiplier=classes_and_constants.back_sell_price_multiplier
    # converter_efficiency=classes_and_constants.converter_efficiency
    # t=classes_and_constants.t
    # battery_capacity=classes_and_constants.battery_capacity
    # inverse_converter_efficiency=classes_and_constants.inverse_converter_efficiency
    # cutoff_cost=classes_and_constants.cutoff_cost
    # cutoff_area=classes_and_constants.cutoff_area

"""
MAIN
"""
def main(_load_forecast_list, _pv_forecast_list, _price,_SoC, best_points_queue):

    get_constants()

    global load_forecast_list, pv_forecast_list,SoC, price

    load_forecast_list =  _load_forecast_list # W
    pv_forecast_list =  _pv_forecast_list # W

    SoC = _SoC # [0,1]
    price =  _price# grid usage price per watt (TODO convert it per energy)

    load_forecast_list = [i*t for i in load_forecast_list] # to energy
    pv_forecast_list = [i*t for i in pv_forecast_list] # to energy

    best_points = []

    triangle = None

    #print('HAS TO BE: ')
    #print(validity_check([10*3600*0.25*converter_efficiency,0,0,-SoC*battery_capacity/3*converter_efficiency,
    #                      10 * 3600 * 0.25*converter_efficiency, 0, 0, -SoC * battery_capacity / 3*converter_efficiency,
    #                      10 * 3600 * 0.25*converter_efficiency, 0, 0, -SoC * battery_capacity / 3*converter_efficiency]))

    # TODO: more particles might need
    for k in range(0, pow(10, int(len(pv_forecast_list)))):
    # for k in range(0, 25):
    # for k in range(0, pow(10, 3)):
        triangle = get_random_triangle()

        while triangle.best.cost > cutoff_cost and triangle.area > cutoff_area:
            new_triangle = update_triangle(triangle)
            if new_triangle is None:
                print('\ntriangle is stucked.')
                break
            triangle = new_triangle
        best_points.append(triangle.best)

    best_point = min(best_points, key=attrgetter('cost'))
    #print('BEST (before point update): ')
    #print(best_point)
    # TODO update problemli best_point = update_point(best_point)
    # worst_point = max(best_points, key=attrgetter('cost'))
    # update_point(worst_point)

    # TODO: add graphic for representation

    # print('\nBEST (simplex): ')
    # print(best_point)

    best_points_queue.put(best_point)
    # return best_point

if __name__ == '__main__':
    get_constants()
    global load_forecast_list, pv_forecast_list, SoC, price
    load_forecast_list =  [9, 9, 9] # W
    pv_forecast_list =  [10, 10, 10] # W
    SoC = 1 # [0,1]
    price = [2,2,2]# grid usage price per watt (TODO convert it per energy)

    # how many parallel process will be created
    process_number=10
    # timeout for each process (in seconds)
    timeout = 2.5

    best_points_queue=Queue()
    best_points_list=[]

    process_list=[]

    killed_process_number=0

    # create process_number many processes
    for x in range(0,process_number):
        process_list.append(Process(target=main,name='main1',args=(load_forecast_list, pv_forecast_list, price , SoC, best_points_queue)))

    # start processes
    for x in range(0,process_number):
        process_list[x].start()

    # wait timeout seconds
    time.sleep(timeout)

    # kill processes that are still run
    for x in range(0,process_number):
        if process_list[x].is_alive():
            killed_process_number+=1
            process_list[x].terminate()
            process_list[x].join()

    print(str(killed_process_number)+' process out of '+str(process_number)+' is killed\n')

    # terminal queue to list
    while not best_points_queue.empty():
        best_points_list.append(best_points_queue.get())

    # get best and print
    point = min(best_points_list, key=attrgetter('cost'))
    print(point)
    print('\nbest cost: '+str(point.cost))
    print('best coordinate'+str(point.coordinate))

    #main([9, 9, 9], [10, 10, 10], [2,2,2], 1)