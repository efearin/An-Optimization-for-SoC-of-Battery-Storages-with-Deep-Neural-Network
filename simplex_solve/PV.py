import random
from operator import attrgetter

# dimention 4 <pvl, pvb, bl, gb>
# pvl: photovoltaic to load | pvb: photovoltaic to battery | bl: battery to load | gb: grid to battery

load_forecast_list = [100, 200, 300, 450]  # W
pv_forecast_list = [300, 200, 150, 40]  # W
battery_capacity = 1  # Ah
battery_voltage = 24  # V
SoC = 0.2
optimal_SoC = 0.4
SoC_mismatch_penalty = 0.001
price = 10
t = 0.25  # h
cutoff_cost = 0.00001
cutoff_area = 0.0001

t = t * 3600  # hour to second
battery_capacity = battery_capacity * battery_voltage * 3600  # to energy


class Point:

    # dimention 4 <pvl, pvb, bl, gb>

    def __init__(self, coordinate, cost=None):
        self.cost = cost
        self.coordinate = coordinate

    def __sub__(self, point):
        summation = 0
        for x in range(0, len(self.coordinate)):
            summation += (self.coordinate[x] - point.coordinate[x]) ** 2
        return summation ** 0.5

    def __str__(self):
        return "pv-l: {} pv-b: {} b-l: {} g-b: {} \ncost: {}".format(self.coordinate[0] / t,
                                                                     self.coordinate[1] / t,
                                                                     self.coordinate[2] / t,
                                                                     self.coordinate[3] / t,
                                                                     self.cost)


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


def validity_check(load_forecast, pv_forecast, coordinate):
    pvc = pv_forecast - (coordinate[1] + coordinate[2])
    gl = load_forecast - (coordinate[2] + coordinate[0])

    if (coordinate[0] < 0 or coordinate[1] < 0 or coordinate[2] < 0 or
            coordinate[3] < 0 or pvc < 0 or gl < 0):
        return False

    new_SoC = (battery_capacity * SoC + t * (coordinate[3] + coordinate[1] - coordinate[2])) / battery_capacity
    if (new_SoC > 1 or new_SoC < 0):
        return False

    return True


def calculate_cost(gl, point: Point):
    new_SoC = (battery_capacity * SoC + t * (
            point.coordinate[3] + point.coordinate[1] - point.coordinate[2])) / battery_capacity
    point.cost = (price * (gl + point.coordinate[3]) + abs(0.4 - new_SoC) * SoC_mismatch_penalty) * t


def get_random_point(initial_coordinate, load_forecast, pv_forecast):
    coordinate_multiplier = 50000
    point_multiplier = 5000

    while True:
        coordinates = []
        for x in range(0, 4):
            if initial_coordinate is None:
                coordinates.append(random.uniform(0, 1) * coordinate_multiplier)
            else:
                coordinates.append(initial_coordinate[x] + random.uniform(0, 1) * point_multiplier)
        if validity_check(load_forecast, pv_forecast, coordinates):
            return Point(coordinates)


def get_random_triangle(load_forecast, pv_forecast):
    points = []

    points.append(get_random_point(None, load_forecast, pv_forecast))
    points.append(get_random_point(points[0].coordinate, load_forecast, pv_forecast))
    while True:
        last_point = get_random_point(points[0].coordinate, load_forecast, pv_forecast)
        if last_point != points[1]:
            points.append(last_point)
            for x in range(0, 3):
                gl = load_forecast - (points[x].coordinate[2] + points[x].coordinate[0])
                calculate_cost(gl, points[x])
            return Triangle(points[0], points[1], points[2])


def update_triangle(triangle, load_forecast, pv_forecast):
    scalars = [2, 1.5, 1.25, 1]

    # move case
    for x in range(0, len(scalars)):
        new_worst_point = triangle.worst.get_point_symmetric_to_points_with_scale(triangle.mid, triangle.best,
                                                                                  scalars[x])
        if validity_check(load_forecast, pv_forecast, new_worst_point.coordinate):
            gl = load_forecast - (new_worst_point.coordinate[2] + new_worst_point.coordinate[0])
            calculate_cost(gl, new_worst_point)
            if new_worst_point.cost < triangle.worst.cost:
                # print('move: ' + str(scalars[x]) + ' ' + str(Triangle(new_worst_point, triangle.mid, triangle.best).worst.cost))
                return Triangle(new_worst_point, triangle.mid, triangle.best)

    # shrink case
    new_worst_point = triangle.worst.get_mid_to_point(triangle.best)
    new_mid_point = triangle.mid.get_mid_to_point(triangle.best)
    if (validity_check(load_forecast, pv_forecast, new_worst_point.coordinate) and
            validity_check(load_forecast, pv_forecast, new_mid_point.coordinate)):
        gl = load_forecast - (new_worst_point.coordinate[2] + new_worst_point.coordinate[0])
        calculate_cost(gl, new_worst_point)
        gl = load_forecast - (new_mid_point.coordinate[2] + new_mid_point.coordinate[0])
        calculate_cost(gl, new_mid_point)
        # print('shrink ' + str(Triangle(triangle.best, new_mid_point, new_worst_point).worst.cost))
        return Triangle(triangle.best, new_mid_point, new_worst_point)

    # triangel stucked
    return None


for w in range(0, len(load_forecast_list)):

    load_forecast = load_forecast_list[w] * t  # to energy
    pv_forecast = pv_forecast_list[w] * t  # to energy

    best_points = []

    triangle = None
    for k in range(0, 1000):
        triangle = get_random_triangle(load_forecast, pv_forecast)
        # print(triangle.best)

        while triangle.best.cost > cutoff_cost and triangle.area > cutoff_area:
            triangle = update_triangle(triangle, load_forecast, pv_forecast)
            if triangle is None:
                print('triangle is stucked.')
                print(triangle.best)
                print(triangle.mid)
                print(triangle.worst)
                break

        best_points.append(triangle.best)

    best_point = min(best_points, key=attrgetter('cost'))
    print('new forecast')
    print(best_point)
