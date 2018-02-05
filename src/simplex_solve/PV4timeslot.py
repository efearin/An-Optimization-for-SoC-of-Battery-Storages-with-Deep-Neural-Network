import random
from operator import attrgetter

# dimention 4 <1pvl, 1pvb, 1bl, 1gb,
#              2pvl, 2pvb, 2bl, 2gb,
#              3pvl, 3pvb, 3bl, 3gb,
#              4pvl, 4pvb, 4bl, 4gb>
# pvl: photovoltaic to load | pvb: photovoltaic to battery | bl: battery to load | gb: grid to battery

load_forecast_list = [100, 200, 300, 450]  # W
pv_forecast_list = [300, 200, 150, 40]  # W
battery_capacity = 1  # Ah
battery_voltage = 24  # V
SoC = 0.2 # [0,1]
optimal_SoC = 0.4 # [0,1]
SoC_mismatch_penalty = 0.001 # [0,1]
price = [10,2,5,4] # grid usage price per watt
t = 0.25  # h
cutoff_cost = 0.00001
cutoff_area = 0.0001

t = t * 3600  # hour to second
load_forecast_list = [i*t for i in load_forecast_list] # to energy
pv_forecast_list = [i*t for i in pv_forecast_list] # to energy
battery_capacity = battery_capacity * battery_voltage * 3600  # to energy


class Point:

    # dimention 4 <1pvl, 1pvb, 1bl, 1gb,
    #              2pvl, 2pvb, 2bl, 2gb,
    #              3pvl, 3pvb, 3bl, 3gb,
    #              4pvl, 4pvb, 4bl, 4gb>

    def __init__(self, coordinate: list, cost=None):
        self.cost = cost
        self.coordinate = coordinate

    def __sub__(self, point):
        summation = 0
        for x in range(0, len(self.coordinate)):
            summation += (self.coordinate[x] - point.coordinate[x]) ** 2
        return summation ** 0.5

    def __str__(self):
        string = ''
        for x in range (0,4):
            string += str(x+1)+"pv-l: "+str(self.coordinate[4*x]/t)+" "+str(x+1)+"pv-b: "+str(self.coordinate[4*x+1]/t)+" "+str(x+1)+"b-l: "+str(self.coordinate[4*x+2]/t)+" "+str(x+1)+"g-b: "+str(self.coordinate[4*x+3]/t)+"\n"
        string += "cost: " + str(self.cost)
        return string

    def get_point_symmetric_to_points_with_scale(self, point1, point2, scale):
        mid_point = point1.get_mid_to_point(point2)
        coordinate = []
        for x in range(0, len(self.coordinate)):
            coordinate.append(self.coordinate[x] + abs(self.coordinate[x] - mid_point.coordinate[x]) * scale)
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
        return (self.best-self.mid) * (self.mid-self.worst) * (self.worst-self.best)


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
        pvc_list.append( pv_forecast_list[l] - (coordinate[4*l+1] + coordinate[4*l+2]))
        gl_list.append(load_forecast_list[l] - (coordinate[4*l+2] + coordinate[4*l]))

        if (coordinate[l] < 0 or coordinate[l+1] < 0 or coordinate[l+2] < 0 or
                coordinate[l+3] < 0 or pvc_list[l] < 0 or gl_list[l] < 0):
            return None

        SoC_list.append((battery_capacity * SoC_list[l] + t * (coordinate[4*l+3] + coordinate[4*l+1] - coordinate[4*l+2])) / battery_capacity)

        if (SoC_list[-1] > 1 or SoC_list[-1] < 0):
            return None

        cost += (price[l] * (gl_list[-1] + coordinate[4*l+3]) + abs(0.4 - SoC_list[-1]) * SoC_mismatch_penalty) * t

    return Point(coordinate, cost)


def get_random_point(initial_coordinate=None):
    """
    :param initial_coordinate: length of 4*load_forecast or None
    :return: if no input given random Point else point close to given coordinate
    """
    coordinate_multiplier = 500
    point_multiplier = 50

    miss_count = 0
    while True:
        coordinates = []
        for x in range(0, len(load_forecast_list*4)):
            if initial_coordinate is None:
                coordinates.append(random.uniform(0, 1) * coordinate_multiplier)
            else:
                coordinates.append(initial_coordinate[x] + random.uniform(0, 1) * point_multiplier)
        point = validity_check(coordinates)
        if point is not None:
            if initial_coordinate is None:
                print('FULLY random point found after '+ str(miss_count)+ ' attempt')
            else:
                print('CLOSE random point found after '+ str(miss_count)+ ' attempt')
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
    scalars = [2, 1.5, 1.25, 1]

    # move case
    for x in range(0, len(scalars)):
        new_worst_point = triangle.worst.get_point_symmetric_to_points_with_scale(triangle.mid, triangle.best,
                                                                                  scalars[x])
        new_worst_point = validity_check(new_worst_point.coordinate)
        if new_worst_point is not None:
            if new_worst_point.cost < triangle.worst.cost:
                print('move: ' + str(scalars[x]) + ' ' + str(Triangle(new_worst_point, triangle.mid, triangle.best).worst.cost))
                return Triangle(new_worst_point, triangle.mid, triangle.best)

    # shrink case
    new_worst_point = triangle.worst.get_mid_to_point(triangle.best)
    new_mid_point = triangle.mid.get_mid_to_point(triangle.best)
    new_worst_point = validity_check(new_worst_point.coordinate)
    new_mid_point = validity_check(new_mid_point.coordinate)
    if (new_worst_point is not None) and (new_mid_point is not None):
        print('shrink ' + str(Triangle(triangle.best, new_mid_point, new_worst_point).worst.cost))
        return Triangle(triangle.best, new_mid_point, new_worst_point)

    # triangel stucked
    return None


best_points = []

triangle = None

for k in range(0, 100):
    triangle = get_random_triangle()
    print(triangle.best)

    while triangle.best.cost > cutoff_cost and triangle.area > cutoff_area:
        triangle = update_triangle(triangle)
        if triangle is None:
            print('triangle is stucked.')
            print(triangle.best)
            print(triangle.mid)
            print(triangle.worst)
            break

    best_points.append(triangle.best)

best_point = min(best_points, key=attrgetter('cost'))
worst_point = max(best_points, key=attrgetter('cost'))
print('BEST: ')
print(best_point)
print('WORST: ')
print(worst_point)