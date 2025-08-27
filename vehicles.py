import random
import pyproj

class Car:
    def __init__(self, lane: int, 
                 position: int, 
                 speed: int=0, 
                 vmax: int=4):
        self.lane = lane
        self.position = position
        self.speed = speed
        self.vmax = vmax
        self.last_lane_change = -100  # шаг последнего перестроения

    def accelerate(self):
        self.speed = min(self.speed + 1, self.vmax)

    def slow_down(self, distance_to_next: int):
        self.speed = min(self.speed, distance_to_next - 1)

    def random_slow_down(self, p: float):
        if self.speed > 0 and random.random() < p:
            self.speed -= 1

    def move(self, road_length: int, direction: int):
        self.position = (self.position + direction * self.speed) % road_length


class Bus(Car):
    def __init__(self, lane, position, speed=0, vmax=3, stops=None, dwell_time=10):
        super().__init__(lane, position, speed, vmax)
        self.stops = stops or []            # позиции остановок (ячейки)
        self.dwell_time = dwell_time        # сколько шагов стоит на остановке
        self.wait_timer = 0
        self.current_stop_index = 0

    def at_stop(self):
        return (self.current_stop_index < len(self.stops)
                and self.position == self.stops[self.current_stop_index])

    def update_stop(self):
        """Возвращает True, если автобус стоит на остановке"""
        if self.at_stop():
            if self.wait_timer < self.dwell_time:
                self.wait_timer += 1
                self.speed = 0
                return True
            else:
                self.wait_timer = 0
                self.current_stop_index += 1
        return False
