import imageio
from matplotlib.patches import FancyBboxPatch, Patch
from PIL import Image
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
from statistics import median, mode
from vehicles import Car, Bus


save_dir = "kasimov_data"
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


def plot_average_speed_steps(avg_speeds: list, 
                             vmax: int, 
                             steps: int, 
                             filename: str = None, 
                             show: bool = True):
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


def draw_road_state(
    road, step: int, filename: str = None,
    bus_entry_cell: int = None, bus_exit_cell: int = None
):
    from matplotlib.patches import Patch, FancyBboxPatch

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
    bus_color = 'limegreen'  # Цвет автобуса

    car_width = 1
    car_height = 0.35

    for car in cars:
        direction = lane_directions[car.lane]
        if isinstance(car, Bus):
            color = bus_color
        else:
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

    # Вход/выход автовокзала
    entry_color = 'purple'
    exit_color = 'red'
    legend_elements = []
    if bus_entry_cell is not None:
        ax.axvline(bus_entry_cell, color=entry_color, linestyle='--', linewidth=2, label='Вход автовокзал')
        legend_elements.append(Patch(facecolor=entry_color, edgecolor='black', label='Вход автовокзал'))
    if bus_exit_cell is not None:
        ax.axvline(bus_exit_cell, color=exit_color, linestyle='--', linewidth=2, label='Выход автовокзал')
        legend_elements.append(Patch(facecolor=exit_color, edgecolor='black', label='Выход автовокзал'))

    if any([lane_direction == -1 for lane_direction in lane_directions]):
        legend_elements += [
            Patch(facecolor=color, edgecolor='black', label=label)
            for direction, color in direction_colors.items()
            for dir_key, label in direction_labels.items()
            if dir_key == direction
        ]
    legend_elements.append(Patch(facecolor=bus_color, edgecolor='black', label='Автобус'))

    if legend_elements:
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


def create_gif(road, terminal, filename='traffic_simulation_with_buses.gif', frames=300, bus_entry_cell=None, bus_exit_cell=None):
    images = []
    for step in range(frames):
        # Спавним автобус, если пришло его время
        terminal.spawn_bus_if_needed(step, road)

        # Обновляем дорогу (теперь с учётом автобусов)
        road.update()

        # Убираем доехавших до терминала
        terminal.collect_finished(road)

        # Рисуем кадр
        frame_file = os.path.join(save_dir, f'frame_{step}.png')
        draw_road_state(road, step, frame_file, bus_entry_cell, bus_exit_cell)
        images.append(Image.open(frame_file))

    # Собираем GIF
    images[0].save(
        os.path.join(save_dir, filename),
        save_all=True,
        append_images=images[1:],
        duration=200,
        loop=0
    )

    # Чистим png-файлы
    for step in range(frames):
        os.remove(os.path.join(save_dir, f'frame_{step}.png'))


def plot_average_speed_density(densities: list, 
                               results: dict, 
                               filename: str = "average_speed_density_2D.png", 
                               show: bool = True):
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


def compare_effect(densities, 
                   avg_speeds_with, 
                   avg_speeds_without, 
                   labels: list, 
                   filename: str, 
                   show: bool = True,
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
    

def get_characteristics_of_path_between_coords(coord1, 
                                               coord2, 
                                               G, 
                                               filename: str="graph.png", 
                                               show: bool=True):
    ox.plot_graph(G)
    
    if not all(('length' in d for _,_,d in G.edges(data=True))):
        G = ox.distance.add_edge_lengths(G)
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    # G = ox.distance.add_edge_lengths(G)

    orig_node = ox.nearest_nodes(G, coord1[1], coord1[0])
    dest_node = ox.nearest_nodes(G, coord2[1], coord2[0])

    try:
        route = nx.shortest_path(G, orig_node, dest_node, weight='length')
    except nx.NetworkXNoPath:
        raise ValueError("Маршрут не найден.")

    fig, ax = ox.plot_graph_route(
        G, route,
        route_color='red',
        route_linewidth=3,
        orig_dest_size=40,
        node_size=5,
        node_color='grey',
        edge_color='lightgray',
        show=False,
        close=False
    )
    
    if filename:
        plt.savefig(os.path.join(save_dir, filename))
    if show:
        plt.show()
    plt.close()


    # Собираем атрибуты по всему маршруту
    edges = []
    for u, v in zip(route[:-1], route[1:]):
        data = G.get_edge_data(u, v)
        # Если мультиграф, берём первый вариант ребра
        if isinstance(data, dict):
            edge_data = list(data.values())[0]
        else:
            edge_data = data
        edges.append(edge_data)
    lengths = [e.get('length', 0) for e in edges if e.get('length', 0) is not None]
    route_length = float(sum(lengths))

    # скорость: возьмем медиану по участкам, где есть скорость
    speeds = [e.get('speed_kph') for e in edges if isinstance(e.get('speed_kph'), (int, float))]
    speed_limit = float(median(speeds)) if speeds else 40.0  # разумный дефолт

    # полосы: мода целочисленных значений, иначе дефолт=2
    lane_vals = []
    for e in edges:
        ln = e.get('lanes')
        if isinstance(ln, str) and ln.isdigit():
            ln = int(ln)
        if isinstance(ln, (int, float)):
            lane_vals.append(int(ln))
    try:
        lanes = mode(lane_vals) if lane_vals else 2
    except:
        lanes = lane_vals[0] if lane_vals else 2

    return G, {
        'length_m': route_length,
        'lanes': lanes,
        'speed_limit_kph': speed_limit,
        'route_nodes': route
    }
    
