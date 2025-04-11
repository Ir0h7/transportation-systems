import matplotlib.pyplot as plt
import numpy as np
import os
import imageio
from matplotlib.patches import FancyBboxPatch, Patch
from PIL import Image


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


def plot_average_speed_steps(avg_speeds: list, vmax: int, steps: int, filename: str = None, show: bool = True):
    plt.figure(figsize=(10, 5))
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


"""def draw_road_state(road, step: int, filename: str = None):
    road_length = road.length
    lanes = road.lanes
    cars = road.cars
    traffic_lights = getattr(road, "traffic_lights", [])

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

    for light in traffic_lights:
        color = 'green' if light.is_green() else 'red'
        for lane in range(lanes):
            plt.plot(light.position, lane + 0.5, 'o', color=color, markersize=12, markeredgecolor='black', zorder=5)

    plt.xlim(0, road_length)
    plt.ylim(-1, lanes)
    plt.title(f'Шаг {step}')
    plt.legend(loc='lower right')
    plt.axis('off')

    if filename:
        plt.savefig(filename)
    else:
        plt.show()

    plt.close()"""



def draw_road_state(road, step: int, filename: str = None):
    road_length = road.length
    lanes = road.lanes
    cars = road.cars
    lane_directions = getattr(road, "lane_directions", [1] * lanes)
    traffic_lights = getattr(road, "traffic_lights", [])

    plt.figure(figsize=(14, lanes))
    ax = plt.gca()
    ax.set_xlim(0, road_length)
    ax.set_ylim(0, lanes)
    ax.fill_between([0, road_length], 0, lanes, color='gray', alpha=0.3)

    for i in range(1, lanes):
        ax.plot([0, road_length], [i] * 2, 'w--', linewidth=1)

    direction_colors = {1: 'blue', -1: 'orange'}
    direction_labels = {1: '→', -1: '←'}

    car_width = 1
    car_height = 0.35

    for car in cars:
        direction = lane_directions[car.lane]
        color = direction_colors.get(direction, 'gray')
        x = car.position - car_width / 2
        y = car.lane + (1 - car_height) / 2
        box = FancyBboxPatch((x, y), car_width, car_height,
                             boxstyle="round,pad=0.02",
                             edgecolor='black', facecolor=color, linewidth=1)
        ax.add_patch(box)

    for light in traffic_lights:
        light_color = 'green' if light.is_green() else 'red'
        for lane in range(lanes):
            ax.plot(light.position, lane + 0.5, 'o', color=light_color,
                    markersize=12, markeredgecolor='black', zorder=5)
            
    if any([lane_direction == -1 for lane_direction in lane_directions]):
        legend_elements = [
            Patch(facecolor=color, edgecolor='black', label=label)
            for direction, color in direction_colors.items()
            for dir_key, label in direction_labels.items()
            if dir_key == direction
        ]
    
        plt.subplots_adjust(bottom=0.15)

        ax.legend(
            handles=legend_elements,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.08),
            ncol=len(legend_elements),
            frameon=False
        )

    plt.title(f'Шаг {step}')
    plt.axis('off')

    if filename:
        plt.savefig(filename)
    else:
        plt.show()

    plt.close()


def create_gif(road, filename='traffic_simulation_2D.gif', frames=20):
    frame_files = []
    
    for i in range(frames):
        frame_path = os.path.join(save_dir, f'frame_{i}.png')
        draw_road_state(road, i, frame_path)
        road.update()
        frame_files.append(frame_path)

    images = [Image.open(f) for f in frame_files]

    gif_path = os.path.join(save_dir, filename)
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        optimize=True,
        duration=2,
        loop=0
    )

    for f in frame_files:
        os.remove(f)

def plot_average_speed_density(densities: list, results: dict, filename: str = "average_speed_density_2D.png", show: bool = True):
    plt.figure(figsize=(10, 5))
    for p, speeds in results.items():
        plt.plot(densities, speeds, label=f"p = {p}")
    plt.xlabel("Плотность потока (машин/км)")
    plt.ylabel("Средняя скорость (км/ч)")
    plt.title("Средняя скорость при разных плотностях и вероятностях торможения")
    plt.legend()
    plt.grid(True)

    if filename:
        plt.savefig(os.path.join(save_dir, filename))
    if show:
        plt.show()
    plt.close()


def visualize_jams(road, filename: str = "jams_2D.png", show: bool = True):
    jam_threshold_speed = 5  # машины со скоростью <= 5 считаются в заторе
    jam_matrix = np.zeros_like(road.road)

    for car in road.cars:
        if car.speed <= jam_threshold_speed:
            jam_matrix[car.lane, car.position] = 1

    plt.figure(figsize=(12, 2))
    plt.imshow(jam_matrix, cmap="Reds", aspect='auto', interpolation='nearest')
    plt.title("Карта дорожных пробок (Красный = Пробка)")
    plt.xlabel("Позиция на дороге")
    plt.ylabel("Полосы")
    plt.colorbar(label="Интенсивность пробок")
    plt.tight_layout()
    
    if filename:
        plt.savefig(os.path.join(save_dir, filename))
    if show:
        plt.show()
    plt.close()


def compare_effect(densities, avg_speeds_with, avg_speeds_without, 
                   labels: list, filename: str, show: bool = True,
                   extra_info: dict = None):
    plt.figure(figsize=(10, 5))
    plt.plot(densities, avg_speeds_with, label=labels[0], marker='o')
    plt.plot(densities, avg_speeds_without, label=labels[1], marker='x')
    plt.xlabel("Плотность потока (машин/км)")
    plt.ylabel("Средняя скорость (км/ч)")
    plt.title(labels[2])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if extra_info: add_extra_info(extra_info)

    if filename:
        plt.savefig(os.path.join(save_dir, filename))
    if show:
        plt.show()
    plt.close()


def add_extra_info(extra_info: dict):
    text_lines = [f"{k}: {v}" for k, v in extra_info.items()]
    info_text = "\n".join(text_lines)
    plt.gcf().text(0.79, 0.75, info_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    
