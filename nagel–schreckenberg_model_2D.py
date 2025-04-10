import random
import numpy as np
from utils import draw_road_state, create_gif, plot_average_speed_density, compare_effect
        

class Car:
    def __init__(self, lane: int, position: int, speed: int = 0, vmax: int = 60):
        self.lane = lane
        self.position = position
        self.speed = speed
        self.vmax = vmax

    def accelerate(self):
        self.speed = min(self.speed + 1, self.vmax)

    def slow_down(self, distance_to_next: int):
        self.speed = min(self.speed, distance_to_next - 1)

    def random_slow_down(self, p: float):
        if self.speed > 0 and random.random() < p:
            self.speed -= 1

    def move(self, road_length: int):
        self.position = (self.position + self.speed) % road_length


class Road2D:
    def __init__(self, road_length: int, num_cars: int, lanes: int = 2, vmax: int = 60, p: float = 0.3,
                 allow_lane_change=True, traffic_lights=None):
        self.length = road_length
        self.lanes = lanes
        self.vmax = vmax
        self.p = p
        self.allow_lane_change = allow_lane_change
        self.road = np.full((lanes, road_length), -1)
        self.cars = []
        self.traffic_lights = traffic_lights or []

        total_cells = lanes * road_length
        positions = random.sample(range(total_cells), num_cars)
        for idx, flat_pos in enumerate(positions):
            lane = flat_pos // road_length
            pos = flat_pos % road_length
            car = Car(lane, pos, random.randint(0, vmax), vmax)
            self.cars.append(car)
            self.road[lane, pos] = idx

    def distance_to_next_car(self, car: Car) -> int:
        for d in range(1, self.length):
            idx = (car.position + d) % self.length
            if self.road[car.lane, idx] != -1:
                return d
        return self.length

    def update(self):
        # Обновляем светофоры
        for light in self.traffic_lights:
            light.update()
    
        # Попытка перестроения
        if self.allow_lane_change:
            for car in self.cars:
                self.attempt_lane_change(car)

        for car in self.cars:
            car.accelerate()
            distance = self.distance_to_next_car(car)
    
            # Проверка на светофор
            red_light_distance = self.is_red_light_ahead(car)
            if red_light_distance is not None:
                distance = min(distance, red_light_distance)
    
            car.slow_down(distance)
            car.random_slow_down(self.p)
    
        # Обновляем позиции машин
        self.road.fill(-1)
        for idx, car in enumerate(self.cars):
            car.move(self.length)
            self.road[car.lane, car.position] = idx

    def get_car_positions(self):
        return np.where(self.road != -1, 1, 0)

    def get_average_speed(self) -> float:
        return np.mean([car.speed for car in self.cars])

    def is_cell_empty(self, lane: int, position: int) -> bool:
        return self.road[lane, position % self.length] == -1

    def can_change_lane(self, car: Car, direction: int) -> bool:
        new_lane = car.lane + direction
        if not (0 <= new_lane < self.lanes):
            return False

        pos = car.position
        # Проверка безопасного перестроения: ячейки впереди и сзади на соседней полосе
        for delta in range(-1, 3):
            check_pos = (pos + delta) % self.length
            if not self.is_cell_empty(new_lane, check_pos):
                return False
        return True

    def attempt_lane_change(self, car: Car):
        distance_current = self.distance_to_next_car(car)

        best_lane = car.lane
        best_distance = distance_current

        for direction in [-1, 1]:
            new_lane = car.lane + direction
            if 0 <= new_lane < self.lanes and self.can_change_lane(car, direction):
                temp_car = Car(new_lane, car.position, car.speed, car.vmax)
                distance_new = self.distance_to_next_car(temp_car)
                if distance_new > best_distance:
                    best_distance = distance_new
                    best_lane = new_lane

        # Перестроение, если выгодно
        if best_lane != car.lane:
            car.lane = best_lane

    def is_red_light_ahead(self, car: Car) -> int | None:
        for light in self.traffic_lights:
            if light.position <= car.position:
                continue

            distance = (light.position - car.position) % self.length
            if light.state == 'red':
                return distance
        return None


class TrafficLight:
    def __init__(self, position: int, cycle=[20, 20]):
        self.position = position
        self.green_time, self.red_time = cycle
        self.state = 'green'
        self.timer = 0

    def update(self):
        self.timer += 1
        if self.state == 'green' and self.timer >= self.green_time:
            self.state = 'red'
            self.timer = 0
        elif self.state == 'red' and self.timer >= self.red_time:
            self.state = 'green'
            self.timer = 0

    def is_green(self) -> bool:
        return self.state == 'green'


steps = 100
road_length = 300
num_cars = 20
lanes = 2
vmax = 60
p = 0.2

traffic_light_position = road_length // 2
light_cycle = [30, 30]


road_gif_1 = Road2D(road_length, num_cars, lanes, vmax, p)
create_gif(road_gif_1, filename='traffic_simulation_without_traffic_lights_2D.gif', frames=steps)

traffic_lights_gif = [TrafficLight(traffic_light_position, light_cycle)]
road_gif_2 = Road2D(road_length, num_cars, lanes, vmax, p, traffic_lights=traffic_lights_gif)
create_gif(road_gif_2, filename='traffic_simulation_with_traffic_lights_2D.gif', frames=steps)


traffic_lights_visualize = [TrafficLight(traffic_light_position, light_cycle)]
road_visualize = Road2D(road_length, num_cars, lanes, vmax, p, traffic_lights=traffic_lights_visualize)

for step in range(steps):
    draw_road_state(road_visualize, step)
    road_visualize.update()


densities = np.linspace(0.03, 0.5, 10)
ps = [0.0, 0.2, 0.5]
results = {}

# Сравнение с перестроением без светофора при разных вероятностях случайного возмущения
for p in ps:
    avg_speeds = []
    for density in densities:
        sum_speed_per_dens = 0
        num_cars = int(road_length * density)
        road_p = Road2D(road_length, num_cars, lanes, vmax, p)
        for _ in range(steps):
            road_p.update()
            sum_speed_per_dens += road_p.get_average_speed()
        avg_speeds.append(sum_speed_per_dens / steps)
    results[p] = avg_speeds

plot_average_speed_density(densities, results)

# Сравнение с и без перестроения (но без светофора)
avg_speeds_with = []
avg_speeds_without = []

for density in densities:
    num_cars = int(road_length * density)

    road_with_lane_changing = Road2D(road_length, num_cars, lanes, vmax, p)
    avg_speed = np.mean([road_with_lane_changing.update() or road_with_lane_changing.get_average_speed() for _ in range(steps)])
    avg_speeds_with.append(avg_speed)

    road_without_lane_changing = Road2D(road_length, num_cars, lanes, vmax, p, allow_lane_change=False)
    avg_speed = np.mean([road_without_lane_changing.update() or road_without_lane_changing.get_average_speed() for _ in range(steps)])
    avg_speeds_without.append(avg_speed)

labels_lane_changing = ['С перестроением', 'Без перестроения', "Сравнение трафика с возможностью перестроения и без"]
compare_effect(densities, avg_speeds_with, avg_speeds_without, labels_lane_changing, "compare_lane_changing_2D.png")


# Сравнение с и без светофора (но с перестроением)
avg_speeds_with_light = []
avg_speeds_without_light = []

for density in densities:
    num_cars = int(road_length * density)
    
    traffic_lights_for_lights = [TrafficLight(traffic_light_position, light_cycle)]
    road_with_traffic_lights = Road2D(road_length, num_cars, lanes, vmax, p, traffic_lights=traffic_lights_for_lights)
    avg_speed = np.mean([road_with_traffic_lights.update() or road_with_traffic_lights.get_average_speed() for _ in range(steps)])
    avg_speeds_with_light.append(avg_speed)

    road_without_traffic_lights = Road2D(road_length, num_cars, lanes, vmax, p)
    avg_speed = np.mean([road_without_traffic_lights.update() or road_without_traffic_lights.get_average_speed() for _ in range(steps)])
    avg_speeds_without_light.append(avg_speed)

labels_traffic_light = ['С светофором', 'Без светофора', "Сравнение трафика с светофором и без"]
compare_effect(densities, avg_speeds_with_light, avg_speeds_without_light, labels_traffic_light, "compare_with_wthout_traffic_lights_2D.png")
