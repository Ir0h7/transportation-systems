import random
import numpy as np
from utils import draw_road_state, create_gif, plot_average_speed_density


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
    def __init__(self, road_length: int, num_cars: int, lanes: int = 2, vmax: int = 60, p: float = 0.3):
        self.length = road_length
        self.lanes = lanes
        self.vmax = vmax
        self.p = p
        self.road = np.full((lanes, road_length), -1)
        self.cars = []

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
        for car in self.cars:
            car.accelerate()
            distance = self.distance_to_next_car(car)
            car.slow_down(distance)
            car.random_slow_down(self.p)

        self.road.fill(-1)
        for idx, car in enumerate(self.cars):
            car.move(self.length)
            self.road[car.lane, car.position] = idx

    def get_car_positions(self):
        return np.where(self.road != -1, 1, 0)

    def get_average_speed(self) -> float:
        return np.mean([car.speed for car in self.cars])


steps = 100
road_length = 300
num_cars = 30
lanes = 2
vmax = 60
p = 0.2

road = Road2D(road_length, num_cars, lanes, vmax, p)

for step in range(steps):
    draw_road_state(road, step)
    road.update()
    
densities = np.linspace(0.03, 0.5, 10)
ps = [0.0, 0.2, 0.5]
results = {}

for p in ps:
    avg_speeds = []
    for density in densities:
        sum_speed_per_dens = 0
        num_cars = int(road_length * density)
        road = Road2D(road_length, num_cars, lanes, vmax, p)
        for _ in range(steps):
            road.update()
            sum_speed_per_dens += road.get_average_speed()
        avg_speeds.append(sum_speed_per_dens / steps)
    results[p] = avg_speeds

plot_average_speed_density(densities, results, filename="average_speed_density.png")

road = Road2D(road_length, num_cars, lanes, vmax, p)
create_gif(road, frames=steps)
