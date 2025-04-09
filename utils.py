import matplotlib.pyplot as plt
import numpy as np
import os
import imageio


save_dir = "images"
os.makedirs(save_dir, exist_ok=True)


def draw_road_state_1D(road, step: int, filename: str = None):
    road_length = road.length
    cars = road.cars

    plt.figure(figsize=(14, 1))
    plt.fill_between([0, road_length], -1, 1, color='gray', alpha=0.3)

    xs = []
    for car in cars:
        xs.append(car.position)

    ys = [0.5] * len(xs)
    plt.plot(xs, ys, 'bs', markersize=10)

    plt.xlim(0, road_length)
    plt.ylim(-1, 1)
    plt.legend(loc='lower right')
    plt.axis('off')

    if filename:
        plt.savefig(filename)
    else:
        plt.show()

    plt.close()
    

def create_gif_1D(road, filename='traffic_simulation_1D.gif', frames=20):
    images = []
    for i in range(frames):
        frame_file = os.path.join(save_dir, f'frame_{i}.png')
        draw_road_state_1D(road, i, frame_file)
        road.update()
        images.append(imageio.imread(frame_file))
    imageio.mimsave(os.path.join(save_dir, filename), images, duration=2, loop=0)
    for i in range(frames):
        os.remove(os.path.join(save_dir, f'frame_{i}.png'))


def plot_heatmap(history: list, filename: str = None, show: bool = True):
    data = np.array(history)
    plt.figure(figsize=(12, 6))
    plt.imshow(data, cmap='Greys', interpolation='nearest', aspect='auto')
    plt.xlabel('Позиция на дороге')
    plt.ylabel('Шаг')
    plt.title('Тепловая карта дорожного потока')
    
    if filename:
        plt.savefig(os.path.join(save_dir, filename))
    if show:
        plt.show()
    plt.close()
    

def plot_average_speed_density(densities: list, results: dict, filename: str = None, show: bool = True):
    plt.figure(figsize=(10, 6))
    for p, speeds in results.items():
        plt.plot(densities, speeds, label=f"p = {p}")
    plt.xlabel("Плотность потока (кол-во машин на ячейку)")
    plt.ylabel("Средняя скорость")
    plt.title("Средняя скорость в зависимости от плотности потока")
    plt.legend()
    plt.grid(True)
    
    if filename:
        plt.savefig(os.path.join(save_dir, filename))
    if show:
        plt.show()
    plt.close()
    

def draw_road_state(road, step: int, filename: str = None):
    road_length = road.length
    lanes = road.lanes
    cars = road.cars

    plt.figure(figsize=(14, lanes))
    plt.fill_between([0, road_length], -1, lanes, color='gray', alpha=0.3)

    for i in range(1, lanes):
        plt.plot([0, road_length], [i] * 2, 'w--', linewidth=1)

    xs_lanes = {i: [] for i in range(lanes)}
    for car in cars:
        xs_lanes[car.lane].append(car.position)

    colors = ['b', 'r', 'g', 'c', 'm', 'y']
    for lane in range(lanes):
        ys = [lane + 0.5] * len(xs_lanes[lane])
        plt.plot(xs_lanes[lane], ys, colors[lane % len(colors)] + 's', markersize=10, label=f'Полоса {lane + 1}')

    plt.xlim(0, road_length)
    plt.ylim(-1, lanes)
    plt.legend(loc='lower right')
    plt.axis('off')

    if filename:
        plt.savefig(filename)
    else:
        plt.show()

    plt.close()


def create_gif(road, filename='traffic_simulation_2D.gif', frames=20):
    images = []
    for i in range(frames):
        frame_file = os.path.join(save_dir, f'frame_{i}.png')
        draw_road_state(road, i, frame_file)
        road.update()
        images.append(imageio.imread(frame_file))
    imageio.mimsave(os.path.join(save_dir, filename), images, duration=2, loop=0)
    for i in range(frames):
        os.remove(os.path.join(save_dir, f'frame_{i}.png'))
        

def plot_average_speed_steps(avg_speeds: list, vmax: int, steps: int, filename: str = None, show: bool = True):
    plt.figure(figsize=(10, 4))
    plt.plot(avg_speeds, label='Средняя скорость')
    plt.axhline(y=vmax, color='red', linestyle='--', label='Максимальная скорость')
    plt.xlabel("Шаг")
    plt.ylabel("Средняя скорость")
    plt.title(f"Средняя скорость за {steps} шагов")
    plt.legend()
    plt.grid()
    
    if filename:
        plt.savefig(os.path.join(save_dir, filename))
    if show:
        plt.show()

    plt.close()
    