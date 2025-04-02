import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
from Cell2D import Cell2D


save_dir = "images"
os.makedirs(save_dir, exist_ok=True)


class TrafficLight:
    def __init__(self, cycle=[5, 5]):
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


class Driver:
    def __init__(self, loc, speed=4, lane=0, direction=1):
        self.loc = loc
        self.speed = speed
        self.lane = lane
        self.direction = direction

    def choose_acceleration(self, dist_to_light, light_state, dist_to_car):
        if light_state == 'red' and dist_to_light < 10:
            return -self.speed
        if dist_to_car < 5:
            return -1
        return 1

    def change_lane(self):
        if self.lane % 2 == 0:
            self.lane += 1
        else:
            self.lane -= 1


class Road(Cell2D):
    def __init__(self, length=100, num_drivers=20):
        self.length = length
        self.drivers = []
        for i in range(num_drivers // 2):
            self.drivers.append(Driver(loc=i * (length // (num_drivers // 2)), lane=i % 2, direction=1))
            self.drivers.append(Driver(loc=length - i * (length // (num_drivers // 2)), lane=2 + (i % 2), direction=-1))
        self.lights = {length // 3: TrafficLight(), 2 * length // 3: TrafficLight()}

    def step(self):
        for light in self.lights.values():
            light.update()

        for driver in self.drivers:
            next_light_pos = min([pos for pos in self.lights.keys() if (pos - driver.loc) * driver.direction > 0],
                                 default=None)
            next_light_dist = abs(next_light_pos - driver.loc) if next_light_pos is not None else self.length
            light_state = self.lights[next_light_pos].state if next_light_pos is not None else 'green'

            next_car = min(
                [d for d in self.drivers if d.lane == driver.lane and (d.loc - driver.loc) * driver.direction > 0],
                key=lambda d: abs(d.loc - driver.loc), default=None)
            dist_to_car = abs(next_car.loc - driver.loc) if next_car else self.length

            acc = driver.choose_acceleration(next_light_dist, light_state, dist_to_car)
            driver.speed = max(0, min(driver.speed + acc, 10))
            driver.loc += driver.speed * driver.direction

            if driver.loc >= self.length:
                driver.loc = 0
            elif driver.loc < 0:
                driver.loc = self.length

            if np.random.rand() < 0.3:
                driver.change_lane()

    def draw(self, filename=None):
        plt.figure(figsize=(12, 4))

        plt.fill_between([0, self.length], -3, 3, color='gray', alpha=0.3, label='Road')
        for light_pos, light in self.lights.items():
            plt.plot([light_pos, light_pos], [-3, 3], 'r' if light.state == 'red' else 'g', linewidth=4)

        for lane in [-2, -1, 1, 2]:
            plt.plot([0, self.length], [lane, lane], 'w--', linewidth=1)

        xs_lanes = {0: [], 1: [], 2: [], 3: []}
        for driver in self.drivers:
            xs_lanes[driver.lane].append(driver.loc)

        plt.plot(xs_lanes[0], [1] * len(xs_lanes[0]), 'bs', markersize=8, label='Lane 0 (→)')
        plt.plot(xs_lanes[1], [2] * len(xs_lanes[1]), 'rs', markersize=8, label='Lane 1 (→)')
        plt.plot(xs_lanes[2], [-1] * len(xs_lanes[2]), 'gs', markersize=8, label='Lane 2 (←)')
        plt.plot(xs_lanes[3], [-2] * len(xs_lanes[3]), 'cs', markersize=8, label='Lane 3 (←)')

        plt.xlim(0, self.length)
        plt.ylim(-3, 3)
        plt.legend()
        plt.axis('off')

        if filename:
            plt.savefig(filename)
        plt.close()

    def create_gif(self, filename='traffic_simulation.gif', frames=20):
        images = []
        for i in range(frames):
            frame_file = os.path.join(save_dir, f'frame_{i}.png')
            self.step()
            self.draw(frame_file)
            images.append(imageio.imread(frame_file))
        imageio.mimsave(os.path.join("images", filename), images, duration=0.2)


road = Road()
road.create_gif()
