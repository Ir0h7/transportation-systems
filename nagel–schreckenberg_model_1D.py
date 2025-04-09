import random
import numpy as np
from utils import draw_road_state_1D, plot_heatmap, create_gif_1D, plot_average_speed_steps


class Car:
    def __init__(self, position: int, speed: int = 0, vmax: int = 60):
        self.position = position
        self.speed = speed
        self.vmax = vmax

    def accelerate(self):
        self.speed = min(self.speed + 1, self.vmax)

    def slow_down(self, distance_to_next):
        self.speed = min(self.speed, distance_to_next - 1)

    def random_slow_down(self, p):
        if self.speed > 0 and random.random() < p:
            self.speed -= 1

    def move(self, road_length):
        self.position = (self.position + self.speed) % road_length


class Road1D:
    def __init__(self, road_length: int, num_cars: int, vmax: int = 60, p: float = 0.3):
        self.length = road_length
        self.vmax = vmax
        self.p = p
        self.road = np.full(road_length, -1)
        self.cars = []

        positions = random.sample(range(road_length), num_cars)
        for idx, pos in enumerate(positions):
            car = Car(position=pos, speed=random.randint(0, vmax), vmax=vmax)
            self.cars.append(car)
            self.road[pos] = idx

    def distance_to_next_car(self, car):
        for d in range(1, self.length):
            idx = (car.position + d) % self.length
            if self.road[idx] != -1:
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
            self.road[car.position] = idx

    def get_car_positions(self):
        return np.where(self.road != -1, 1, 0)

    def get_average_speed(self):
        return np.mean([car.speed for car in self.cars])


steps = 100
road_length = 300
num_cars = 10
vmax = 60
p = 0.3

road = Road1D(road_length, num_cars, vmax, p)
history = []
avg_speeds = []

for step in range(steps):
    draw_road_state_1D(road, step)
    history.append(road.get_car_positions())
    road.update()
    avg_speeds.append(road.get_average_speed())

road = Road1D(road_length, num_cars, vmax, p)
create_gif_1D(road, frames=steps)

plot_heatmap(history, "traffic_heatmap_1D.png")

plot_average_speed_steps(avg_speeds, vmax, steps, filename="average_speed_steps_1D.png")
