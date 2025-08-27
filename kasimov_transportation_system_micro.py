import random
import numpy as np
from utils_kasimov import draw_road_state, create_gif, plot_average_speed_density, compare_effect, \
                    get_characteristics_of_path_between_coords
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import colors, cm
import numpy as np
from matplotlib.lines import Line2D
from shapely.geometry import Point
from geopy.distance import geodesic
import os
import contextily as cx
from vehicles import Car, Bus


save_dir = "kasimov_data"

class CityStreetGraph:
    def __init__(self, city_name: str = None, network_type: str = "drive"):
        self.city_name = city_name
        self.network_type = network_type
        self.graph = None              # Географический (WGS84, градусы)
        self.graph_proj = None         # Проектированный (метры), для консолидации и метрик
        self.highlighted_nodes = set()

    # ---------- Загрузка ----------
    def load_graph_by_place(self):
        if not self.city_name:
            raise ValueError("Название города не указано.")
        self.graph = ox.graph_from_place(self.city_name, network_type=self.network_type, simplify=True)
        self.graph_proj = None  # сбрасываем проектированную версию, т.к. граф обновился

    def load_graph_by_point(self, lat: float, lon: float, distance: int = 3000):
        self.graph = ox.graph_from_point((lat, lon), dist=distance, network_type=self.network_type, simplify=True)
        self.graph_proj = None  # сброс

    # ---------- Проекция и консолидация ----------
    def project_graph(self, to_crs=None):
        """
        Проецирует граф: если to_crs=None — в подходящую метрическую проекцию (обычно UTM).
        Результат кладётся в self.graph_proj.
        """
        if self.graph is None:
            raise ValueError("Граф ещё не загружен.")
        self.graph_proj = ox.project_graph(self.graph, to_crs=to_crs)

    def consolidate_intersections(self, tolerance: float = 15, dead_ends: bool = True):
        """
        Объединяет близко расположенные узлы перекрёстков (в метрах).
        Работает ТОЛЬКО на проектированном графе.
        После консолидации обновляет обе версии графа: graph_proj и graph (WGS84).
        Параметр tolerance ~ 10–25 м обычно даёт хороший результат.
        """
        if self.graph is None:
            raise ValueError("Граф ещё не загружен.")
        if self.graph_proj is None:
            # гарантируем проектирование
            self.project_graph()

        # В разных версиях osmnx возвращаемое значение может отличаться
        res = ox.simplification.consolidate_intersections(
            self.graph_proj, tolerance=tolerance, dead_ends=dead_ends
        )
        Gc = res[0] if isinstance(res, tuple) else res

        # сохраняем проектированный граф после консолидации
        self.graph_proj = Gc
        # и обновляем географический граф, чтобы все остальные функции работали как раньше
        self.graph = ox.project_graph(self.graph_proj, to_crs="EPSG:4326")

    # ---------- Инфо ----------
    def show_basic_info(self):
        if self.graph is None:
            raise ValueError("Граф ещё не загружен.")
        stats = ox.basic_stats(self.graph)
        for k, v in stats.items():
            print(f"{k}: {v}")

    def show_basic_info_extended(self):
        """
        Расширенная статистика по площади выпуклой оболочки ребер.
        Использует проектированную версию графа (если нет — создаёт).
        """
        if self.graph is None:
            raise ValueError("Граф ещё не загружен.")
        if self.graph_proj is None:
            self.project_graph()

        edges = ox.graph_to_gdfs(self.graph_proj, nodes=False, edges=True)
        convex_hull = edges.geometry.union_all().convex_hull
        stats = ox.basic_stats(self.graph_proj, area=convex_hull.area)
        for k, v in stats.items():
            print(f"{k}: {v}")

    # ---------- Визуализация ----------
    def draw_graph(self, save_path: str = None):
        if self.graph is None:
            raise ValueError("Граф ещё не загружен.")
        fig, ax = ox.plot_graph(self.graph, node_size=5, edge_color='gray')
        if save_path:
            fig.savefig(save_path, dpi=300)
            print(f"Граф сохранён в файл: {save_path}")
        else:
            plt.show()

    def draw_graph_with_highlights(self):
        fig, ax = ox.plot_graph(self.graph, node_size=5, edge_color="gray", show=False, close=False)
        if self.highlighted_nodes:
            gdf_nodes = ox.graph_to_gdfs(self.graph, edges=False)
            highlighted = gdf_nodes.loc[list(self.highlighted_nodes)]
            ax.scatter(highlighted.geometry.x, highlighted.geometry.y, s=30, c='red', edgecolors='white', zorder=5)
        plt.show()

    # ---------- Сохранение/загрузка ----------
    def save_graphml(self, filename: str = "kasimov.graphml"):
        if self.graph is None:
            raise ValueError("Граф ещё не загружен.")
        filepath = os.path.join(save_dir, filename)
        ox.save_graphml(self.graph, filepath)

    def load_graphml(self, filename: str = "kasimov.graphml"):
        filepath = os.path.join(save_dir, filename)
        self.graph = ox.load_graphml(filepath)
        self.graph_proj = None  # сбрасываем: при необходимости пересчитаем

    # ---------- Утилиты ----------
    def highlight_nodes_near_locations(self, locations: list[tuple[float, float, float]]):
        """
        Подсвечивает узлы в радиусе radius_m от (lat, lon).
        """
        if self.graph is None:
            raise ValueError("Граф ещё не загружен.")
        self.highlighted_nodes.clear()
        for lat, lon, radius in locations:
            for node_id, data in self.graph.nodes(data=True):
                dist = geodesic((lat, lon), (data['y'], data['x'])).meters
                if dist <= radius:
                    self.highlighted_nodes.add(node_id)
        self.draw_graph_with_highlights()

    def remove_highlighted_nodes(self, save_filename: str = None):
        """
        Удаляет подсвеченные узлы из обеих версий графа (если proj существует),
        чтобы они не «рассинхронизировались».
        """
        if self.graph is None:
            raise ValueError("Граф ещё не загружен.")
        # удаляем из географического графа
        self.graph.remove_nodes_from(self.highlighted_nodes)
        # удаляем из проектированного, если он уже построен
        if self.graph_proj is not None:
            existing = [n for n in self.highlighted_nodes if n in self.graph_proj.nodes]
            self.graph_proj.remove_nodes_from(existing)

        self.highlighted_nodes.clear()

        if save_filename:
            filepath = os.path.join(save_dir, save_filename)
            ox.save_graphml(self.graph, filepath)

    def add_node(self, lat: float, lon: float):
        if self.graph is None:
            raise ValueError("Граф ещё не загружен.")
        new_node_id = max(self.graph.nodes) + 1
        self.graph.add_node(new_node_id, x=lon, y=lat)

        nearest = ox.distance.nearest_nodes(self.graph, X=lon, Y=lat)
        distance = ox.distance.great_circle_vec(lat, lon, self.graph.nodes[nearest]['y'], self.graph.nodes[nearest]['x'])
        self.graph.add_edge(new_node_id, nearest, length=distance)
        self.graph.add_edge(nearest, new_node_id, length=distance)

        print(f"Добавлен узел {new_node_id}, соединён с узлом {nearest}.")
        fig, ax = ox.plot_graph(self.graph, node_size=5, edge_color="gray", show=False, close=False)
        ax.scatter(lon, lat, s=60, c='lime', edgecolors='black', zorder=5)
        plt.show()

    def find_centralities(self):
        line_G = nx.line_graph(self.graph)
        print("Центральность по степени:")
        edge_centrality = nx.degree_centrality(line_G)
        self.show_centrality_heatmap(edge_centrality)

        print("Центральность по близости:")
        edge_centrality = nx.closeness_centrality(line_G)
        self.show_centrality_heatmap(edge_centrality)

    def show_centrality_heatmap(self, edge_centrality, highlight_top_nodes=True):
        if self.graph is None:
            raise ValueError("Граф ещё не загружен.")
        ev = []
        for edge in self.graph.edges(keys=False):
            edge_key = edge + (0,)
            val = edge_centrality.get(edge_key, 0)
            ev.append(val)

        norm = colors.Normalize(vmin=min(ev) * 0.8, vmax=max(ev))
        cmap = cm.ScalarMappable(norm=norm, cmap=cm.inferno)
        ec = [cmap.to_rgba(val) for val in ev]

        fig, ax = ox.plot_graph(
            self.graph,
            bgcolor='black',
            node_size=0,
            edge_color=ec,
            edge_linewidth=2,
            edge_alpha=1,
            show=False,
            close=False
        )
        # Добавляем подложку карты
        try:
            cx.add_basemap(ax, source=cx.providers.CartoDB.Voyager)
        except Exception as e:
            print(f"Не удалось добавить подложку: {e}")

        if highlight_top_nodes:
            node_centrality = nx.degree_centrality(self.graph)
            top_nodes = sorted(node_centrality.items(), key=lambda x: x[1], reverse=True)[:30]
            gdf_nodes = ox.graph_to_gdfs(self.graph, edges=False)
            for node_id, score in top_nodes:
                geom = gdf_nodes.loc[node_id].geometry
                ax.scatter(geom.x, geom.y, s=50, c='cyan', edgecolors='white', zorder=5)

        cbar = plt.colorbar(cmap, ax=ax, fraction=0.03, pad=0.02)
        cbar.ax.yaxis.set_tick_params(color='white')

        bus_station_lat = 54.946817
        bus_station_lon = 41.409804
        ax.scatter(bus_station_lon, bus_station_lat, marker='s', s=90, color='dodgerblue', edgecolor='black', label='Автовокзал')
        ax.legend(loc='upper left')

        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        ax.set_axis_off()
        plt.show()

    def analyze_station_metrics(self, station_coords, speed_kmh=30):
        print("\nМетрики для вокзала:")
        station_node = ox.distance.nearest_nodes(self.graph, X=station_coords[1], Y=station_coords[0])

        lengths = nx.single_source_dijkstra_path_length(self.graph, station_node, weight="length")
        avg_distance_m = np.mean(list(lengths.values()))
        avg_time_min = avg_distance_m / (speed_kmh * 1000 / 60)
        print(f"Среднее время в пути: {avg_time_min:.1f} мин")

        time_10_m = (10 / 60) * speed_kmh * 1000
        time_15_m = (15 / 60) * speed_kmh * 1000

        reachable_10 = [n for n, d in lengths.items() if d <= time_10_m]
        reachable_15 = [n for n, d in lengths.items() if d <= time_15_m]

        max_dist = max(lengths.values()) if lengths else 0
        print(f"Макс. расстояние: {max_dist/1000:.2f} км (~{max_dist/(speed_kmh * 1000 / 60):.1f} мин)")

        node_colors = []
        for node in self.graph.nodes:
            if node in reachable_10:
                node_colors.append("lime")
            elif node in reachable_15:
                node_colors.append("orange")
            else:
                node_colors.append("lightgray")

        fig, ax = ox.plot_graph(
            self.graph,
            node_color=node_colors,
            node_size=8,
            edge_color="lightgray",
            edge_linewidth=0.8,
            bgcolor="white",
            show=False,
            close=False
        )
        # Добавляем подложку карты
        try:
            cx.add_basemap(ax, source=cx.providers.CartoDB.Voyager)
        except Exception as e:
            print(f"Не удалось добавить подложку: {e}")

        bus_station_lat = 54.946817
        bus_station_lon = 41.409804
        ax.scatter(bus_station_lon, bus_station_lat, marker='s', s=90, color='dodgerblue', edgecolor='black', label='Автовокзал')

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='До 10 мин', markerfacecolor='lime', markersize=8),
            Line2D([0], [0], marker='o', color='w', label='До 15 мин', markerfacecolor='orange', markersize=8),
            Line2D([0], [0], marker='o', color='w', label='>15 мин', markerfacecolor='lightgray', markersize=8),
        ]
        star_proxy = plt.scatter([], [], marker='s', s=90, color='dodgerblue', edgecolor='black', label='Автовокзал')
        # Добавляем proxy-artist к списку
        legend_elements.append(star_proxy)
        ax.legend(handles=legend_elements, loc='upper left')
        ax.set_axis_off()
        plt.show()

    def find_bottlenecks(self, top_k=50):
        print("\nПоиск транспортных проблем:")
        simple_graph = nx.DiGraph()
        for u, v, data in self.graph.edges(data=True):
            length = data.get("length", 1)
            if simple_graph.has_edge(u, v):
                simple_graph[u][v]["length"] = min(simple_graph[u][v]["length"], length)
            else:
                simple_graph.add_edge(u, v, length=length)

        edge_betweenness = nx.edge_betweenness_centrality(simple_graph, weight="length")

        edge_scores = []
        for u, v, k in self.graph.edges(keys=True):
            edge = (u, v, k)
            btwn = edge_betweenness.get((u, v), 0) or edge_betweenness.get((v, u), 0)
            length = self.graph[u][v][k].get("length", 1)
            deg_u = self.graph.degree(u)
            deg_v = self.graph.degree(v)
            min_deg = min(deg_u, deg_v)
            score = (btwn / length) * (1 / (min_deg + 0.1))
            edge_scores.append((edge, score, btwn, length, min_deg))

        edge_scores.sort(key=lambda x: x[1], reverse=True)
        worst_edges = [edge for edge, *_ in edge_scores[:top_k]]

        edge_colors = []
        for edge in self.graph.edges(keys=True):
            edge_colors.append("red" if edge in worst_edges else "gray")

        fig, ax = ox.plot_graph(
            self.graph,
            edge_color=edge_colors,
            edge_linewidth=2.8,
            node_size=0,
            edge_alpha=0.9,
            bgcolor="white",
            show=False,
            close=False
        )
        plt.show()

        print("Топ проблемных участков:")
        for i, (edge, score, btwn, length, min_deg) in enumerate(edge_scores[:top_k]):
            u, v, k = edge
            street_name = self.graph[u][v][k].get("name", "Без названия")
            print(
                f"{i+1}. {street_name} ({u} → {v}) | Score: {score:.4f} | "
                f"Betweenness: {btwn:.5f} | Длина: {length:.1f} м | Мин. степень: {min_deg}"
            )


kasimov = CityStreetGraph(network_type="drive_service")

kasimov.load_graphml("kasimov_new_drive_service_cleaned.graphml")

kasimov.project_graph()


class Road2D:
    def __init__(self, road_length: int, 
                 num_cars: int, 
                 lanes: int=2, 
                 vmax: int=4, 
                 p: float=0.2,
                 allow_lane_change: bool=True, 
                 lane_directions: list[int]=None, 
                 traffic_lights=None,
                 cell_length: float=7.5):
        self.length = road_length  # в ячейках
        self.cell_length = cell_length  # длина одной ячейки в метрах
        self.lanes = lanes
        self.lane_directions = lane_directions or [1] * lanes
        self.time_step = 0
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
        direction = self.lane_directions[car.lane]
        for d in range(1, self.length):
            idx = (car.position + direction * d) % self.length
            if self.road[car.lane, idx] != -1:
                return d
        return self.length

    def update(self):
        # Обновляем светофоры
        for light in self.traffic_lights:
            light.update()

        # Попытка перестроения (только для машин и автобусов в движении)
        if self.allow_lane_change:
            for car in self.cars:
                if isinstance(car, Bus) and car.update_stop():
                    continue  # автобус стоит — не перестраивается
                if self.time_step - car.last_lane_change > 5:
                    self.attempt_lane_change(car)

        self.time_step += 1

        # Шаг обновления скоростей
        for car in self.cars:
            if isinstance(car, Bus) and car.update_stop():
                continue  # автобус стоит
            car.accelerate()
            distance = self.distance_to_next_car(car)

            # Проверка светофора
            red_light_distance = self.is_red_light_ahead(car)
            if red_light_distance is not None:
                distance = min(distance, red_light_distance)

            car.slow_down(distance)
            car.random_slow_down(self.p)

        # Пересчёт позиций
        self.road.fill(-1)
        for idx, car in enumerate(self.cars):
            if isinstance(car, Bus) and car.update_stop():
                self.road[car.lane, car.position] = idx
                continue
            car.move(self.length, self.lane_directions[car.lane])
            self.road[car.lane, car.position] = idx

    def get_car_positions(self):
        return np.where(self.road != -1, 1, 0)

    def get_average_speed(self) -> float:
        return np.mean([car.speed for car in self.cars])

    def is_cell_empty(self, lane: int, position: int) -> bool:
        return self.road[lane, position % self.length] == -1

    def can_change_lane(self, car: Car, direction: int) -> bool:
        new_lane = car.lane + direction
        
        if self.lane_directions[car.lane] != self.lane_directions[new_lane]:
            return False

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
    
                if distance_new > best_distance + 2:
                    best_distance = distance_new
                    best_lane = new_lane
    
        # Перестроение, если выгодно
        if best_lane != car.lane and random.random() < 0.7:
            car.lane = best_lane
            car.last_lane_change = self.time_step

    def is_red_light_ahead(self, car: Car) -> int | None:
        for light in self.traffic_lights:
            if light.position <= car.position:
                continue

            distance = (light.position - car.position) % self.length
            if light.state == 'red':
                return distance
        return None
    
    def get_average_speed_kmh(self) -> float:
        avg_speed_cells = self.get_average_speed()
        meters_per_step = avg_speed_cells * self.cell_length
        return meters_per_step * 3.6  # м/с -> км/ч

    def get_density_per_km(self) -> float:
        total_road_m = self.length * self.cell_length
        return len(self.cars) / (total_road_m / 1000)
    
    def road2d_update_with_buses(self):
        # Обновляем светофоры
        for light in self.traffic_lights:
            light.update()

        # Попытка перестроения (только для машин и автобусов в движении)
        if self.allow_lane_change:
            for car in self.cars:
                if isinstance(car, Bus) and car.update_stop():
                    continue  # автобус стоит — не перестраивается
                if self.time_step - car.last_lane_change > 5:
                    self.attempt_lane_change(car)

        self.time_step += 1

        # Шаг обновления скоростей
        for car in self.cars:
            if isinstance(car, Bus) and car.update_stop():
                continue  # автобус стоит
            car.accelerate()
            distance = self.distance_to_next_car(car)

            # Проверка светофора
            red_light_distance = self.is_red_light_ahead(car)
            if red_light_distance is not None:
                distance = min(distance, red_light_distance)

            car.slow_down(distance)
            car.random_slow_down(self.p)

        # Пересчёт позиций
        self.road.fill(-1)
        for idx, car in enumerate(self.cars):
            if isinstance(car, Bus) and car.update_stop():
                self.road[car.lane, car.position] = idx
                continue
            car.move(self.length, self.lane_directions[car.lane])
            self.road[car.lane, car.position] = idx


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
    

class BusTerminal:
    def __init__(self, entry_position, exit_position, schedule, dwell_time=15):
        """
        entry_position, exit_position: ячейки дороги
        schedule: список шагов симуляции, когда выпускать автобусы
        """
        self.entry_position = entry_position
        self.exit_position = exit_position
        self.schedule = schedule
        self.spawned = []
        self.finished = []
        self.dwell_time = dwell_time

    def spawn_bus_if_needed(self, step, road: "Road2D"):
        """Выпустить автобус, если сейчас его время по расписанию"""
        if step in self.schedule:
            bus = Bus(lane=0, position=self.entry_position,
                      vmax=3, stops=[self.exit_position],
                      dwell_time=self.dwell_time)
            road.cars.append(bus)
            road.road[bus.lane, self.entry_position] = len(road.cars) - 1
            self.spawned.append(bus)
            return bus
        return None

    def collect_finished(self, road: "Road2D"):
        """Убрать автобусы, которые доехали до терминала"""
        for bus in list(road.cars):
            if isinstance(bus, Bus) and bus.position == self.exit_position:
                self.finished.append(bus)
                road.cars.remove(bus)



def get_average_speed_with_densities(
    densities, 
    steps: int=200, 
    road_length: int=300, 
    vmax: int=4, 
    p: float=0.2,
    cell_length: float=7.5, 
    lanes: int=2, 
    lane_directions: list[int]=None,
    lane_changing: bool=False,
    traffic_light: bool=False,
    traffic_light_positions=None,
    light_cycle=None
):
    avg_speeds_with = []
    avg_speeds_without = []

    for density in densities:
        num_cars = int(density * road_length * cell_length / 1000)

        if lane_changing:
            road_with = Road2D(road_length, num_cars, lanes, vmax, p)
            road_without = Road2D(road_length, num_cars, lanes, vmax, p, allow_lane_change=False)
        elif traffic_light:
            # создаём светофоры только если переданы позиции и цикл
            if traffic_light_positions is not None and light_cycle is not None:
                traffic_lights = [TrafficLight(pos, light_cycle) for pos in traffic_light_positions]
            else:
                traffic_lights = []
            road_with = Road2D(road_length, num_cars, lanes, vmax, p, traffic_lights=traffic_lights)
            road_without = Road2D(road_length, num_cars, lanes, vmax, p)
        else:
            road_with = Road2D(road_length, num_cars, lanes, vmax, p)
            road_without = Road2D(road_length, num_cars, lanes, vmax, p)

        avg_speed = np.mean([road_with.update() or road_with.get_average_speed_kmh() for _ in range(steps)])
        avg_speeds_with.append(avg_speed)

        avg_speed = np.mean([road_without.update() or road_without.get_average_speed_kmh() for _ in range(steps)])
        avg_speeds_without.append(avg_speed)

    return avg_speeds_with, avg_speeds_without


def plot_route_with_bus_stations_and_lights(G, orig, dest, old_bus_in, old_bus_out, new_bus_in, new_bus_out):
    # ищем ближайшие узлы
    orig_node = ox.distance.nearest_nodes(G, orig[1], orig[0])
    dest_node = ox.distance.nearest_nodes(G, dest[1], dest[0])
    old_in_node = ox.distance.nearest_nodes(G, old_bus_in[1], old_bus_in[0])
    old_out_node = ox.distance.nearest_nodes(G, old_bus_out[1], old_bus_out[0])
    # new_in_node = ox.distance.nearest_nodes(G, new_bus_in[1], new_bus_in[0])
    # new_out_node = ox.distance.nearest_nodes(G, new_bus_out[1], new_bus_out[0])

    # маршрут
    route = nx.shortest_path(G, orig_node, dest_node, weight="length")

    # ========== Авто-определение светофоров ==========
    traffic_lights = []
    for n in route:
        node_data = G.nodes[n]
        if "highway" in node_data and node_data["highway"] == "traffic_signals":
            traffic_lights.append(n)

    # --- Plot the full graph as background ---
    fig, ax = ox.plot_graph(G, node_size=5, edge_color='lightgray', bgcolor='white', show=False, close=False, figsize=(12, 8))

    # Overlay the route
    route_xs = [G.nodes[n]['x'] for n in route]
    route_ys = [G.nodes[n]['y'] for n in route]
    ax.plot(route_xs, route_ys, color='black', linewidth=3, zorder=3, label='Маршрут')

    # координаты для маркеров
    nodes_coords = {
        "Начало маршрута": orig_node,
        "Конец маршрута": dest_node,
        "Текущий автовокзал (въезд)": old_in_node,
        "Текущий автовокзал (выезд)": old_out_node,
        # "Новый автовокзал (въезд)": new_in_node,
        # "Новый автовокзал (выезд)": new_out_node,
    }

    colors = {
        "Начало маршрута": "green",
        "Конец маршрута": "red",
        "Текущий автовокзал (въезд)": "blue",
        "Текущий автовокзал (выезд)": "blue",
        # "Новый автовокзал (въезд)": "orange",
        # "Новый автовокзал (выезд)": "orange",
    }

    markers = {
        "Начало маршрута": "o",
        "Конец маршрута": "o",
        "Текущий автовокзал (въезд)": "s",
        "Текущий автовокзал (выезд)": "s",
        # "Новый автовокзал (въезд)": "D",
        # "Новый автовокзал (выезд)": "D",
    }

    legend_elements = []
    plotted_labels = set()
    for label, node in nodes_coords.items():
        x, y = G.nodes[node]['x'], G.nodes[node]['y']
        m = markers[label]
        c = colors[label]
        ax.scatter(x, y, c=c, s=80, marker=m, edgecolors='black', zorder=5, label=label)
        if label not in plotted_labels:
            legend_elements.append(Line2D([0], [0], marker=m, color='w', label=label, markerfacecolor=c, markeredgecolor='black', markersize=10))
            plotted_labels.add(label)

    # Светофоры
    for i, node in enumerate(traffic_lights):
        x, y = G.nodes[node]['x'], G.nodes[node]['y']
        ax.scatter(x, y, c="yellow", s=120, edgecolors="black", marker="*", zorder=6, label="Светофор" if i == 0 else None)
    if traffic_lights:
        legend_elements.append(Line2D([0], [0], marker='*', color='w', label='Светофор', markerfacecolor='yellow', markeredgecolor='black', markersize=15))

    # Маршрут
    legend_elements.append(Line2D([0], [0], color='black', lw=3, label='Маршрут'))

    ax.legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

    return route, traffic_lights


def map_coords_to_road(route, traffic_lights, G, cell_length=7.5):
    """
    Преобразуем светофоры из OSM в позиции вдоль 1D дороги (в ячейках).
    """
    # Длина вдоль маршрута от начала
    cumulative_length = {route[0]: 0.0}
    total = 0.0
    for u, v in zip(route[:-1], route[1:]):
        edge_data = list(G.get_edge_data(u, v).values())[0]
        edge_len = edge_data.get("length", 0)
        total += edge_len
        cumulative_length[v] = total

    # Проецируем светофоры на эту шкалу
    positions = []
    for tl in traffic_lights:
        if tl in cumulative_length:
            dist_m = cumulative_length[tl]
            pos = int(dist_m / cell_length)
            positions.append(pos)

    return positions



steps = 1200
road_length = 300
num_cars = 20
lanes = 2
lane_directions = [1, 1, -1, -1]
vmax = 4
p = 0.2
cell_length = 7.5

vmax_kmh = int(vmax * cell_length * 3.6)
road_length_km = road_length * cell_length / 1000

filename_lane_changing = "compare_with_without_lane_changing_2D.png"
filename_traffic_light = "compare_with_without_traffic_lights_2D.png"

""" Моделирование реального участка Советской улицы """
# Составляем расписание (35 автобусов за сутки = каждые ~200 шагов при 7200 шагах)
day_steps = steps  # у тебя steps=300, но для суток лучше steps=7200
num_buses = 35
interval = max(1, day_steps // num_buses)  
bus_schedule = [i * interval for i in range(num_buses) if i * interval < day_steps]

orig_coord = (54.937270, 41.391325)
dest_coord = (54.957764, 41.428500)

old_bus_in = (54.947195, 41.409939)
old_bus_out = (54.946788, 41.409199)

new_bus_in = (54.954275, 41.422362)
new_bus_out = (54.954501, 41.422771)

# Получаем маршрут и светофоры с карты
route, traffic_lights = plot_route_with_bus_stations_and_lights(
    kasimov.graph, orig_coord, dest_coord, old_bus_in, old_bus_out, new_bus_in, new_bus_out
)

# Характеристики дороги
G, road_data = get_characteristics_of_path_between_coords(orig_coord, dest_coord, kasimov.graph)

if road_data:
    road_length = int(road_data['length_m'] / cell_length)
    road_length_km = road_data['length_m'] / 1000
    lanes = road_data['lanes']
    vmax = int((road_data['speed_limit_kph'] / 3.6) / cell_length)
    vmax_kmh = int(road_data['speed_limit_kph'])

    extra_info = {
        "Макс. скорость": f"{vmax_kmh} км/ч",
        "Длина дороги": f"{road_length_km:.2f} км",
        "Количество полос": lanes,
    }

    # реальные позиции светофоров
    traffic_light_positions = map_coords_to_road(route, traffic_lights, G, cell_length=cell_length)
    
    # цикл работы светофоров (можно усложнить и хранить разные)
    light_cycle = [80, 80]

    # Получаем id узлов для входа/выхода автовокзала
    entry_node = ox.distance.nearest_nodes(G, old_bus_in[1], old_bus_in[0])
    exit_node = ox.distance.nearest_nodes(G, old_bus_out[1], old_bus_out[0])

    # Получаем позиции вдоль маршрута (в ячейках)
    entry_cell = map_coords_to_road(route, [entry_node], G, cell_length=cell_length)[0]
    exit_cell = map_coords_to_road(route, [exit_node], G, cell_length=cell_length)[0]

    terminal = BusTerminal(entry_cell, exit_cell, schedule=bus_schedule, dwell_time=15)



densities = np.linspace(3, 200, 20)

# Сравнение с и без перестроения (но без светофора)
avg_speeds_with_lane_changing, avg_speeds_without_lane_changing = get_average_speed_with_densities(densities, 
                                                                                 steps, 
                                                                                 road_length, 
                                                                                 vmax, 
                                                                                 cell_length=cell_length, 
                                                                                 lanes=lanes,
                                                                                 lane_changing=True)

labels_lane_changing = ['С перестроением', 'Без перестроения', "Сравнение трафика с возможностью перестроения и без"]
compare_effect(densities, 
               avg_speeds_with_lane_changing, 
               avg_speeds_without_lane_changing, 
               labels_lane_changing, 
               filename=filename_lane_changing, 
               extra_info=extra_info)

# Сравнение с и без светофора (но с перестроением)
avg_speeds_with_light, avg_speeds_without_light = get_average_speed_with_densities(densities, 
                                                                                 steps, 
                                                                                 road_length, 
                                                                                 vmax, 
                                                                                 cell_length=cell_length, 
                                                                                 lanes=lanes,
                                                                                 traffic_light_positions=traffic_light_positions,
                                                                                 light_cycle=light_cycle)

labels_traffic_light = ['С светофором', 'Без светофора', "Сравнение трафика с светофором и без"]
compare_effect(densities, 
               avg_speeds_with_light, 
               avg_speeds_without_light, 
               labels_traffic_light, 
               filename=filename_traffic_light,
               extra_info=extra_info)


road = Road2D(road_length, num_cars, lanes, vmax, p,
            traffic_lights=[TrafficLight(pos, light_cycle) for pos in traffic_light_positions])

create_gif(
    road, terminal,
    filename='traffic_simulation_with_traffic_lights_and_buses.gif',
    frames=steps,
    bus_entry_cell=entry_cell,
    bus_exit_cell=exit_cell
)

"""road_gif_1 = Road2D(road_length, num_cars, lanes, vmax, p)
create_gif(road_gif_1, filename='traffic_simulation_without_traffic_lights_2D.gif', frames=steps)

road_gif_2 = Road2D(road_length, num_cars, lanes=4, vmax=vmax, p=p, lane_directions=lane_directions)
create_gif(road_gif_2, filename='traffic_simulation_without_traffic_lights_4-lanes_2D.gif', frames=steps)

traffic_lights_gif = [TrafficLight(traffic_light_position, light_cycle)]
road_gif_3 = Road2D(road_length, num_cars, lanes, vmax, p, traffic_lights=traffic_lights_gif)
create_gif(road_gif_3, filename='traffic_simulation_with_traffic_lights_2D.gif', frames=steps)


traffic_lights_visualize = [TrafficLight(traffic_light_position, light_cycle)]
road_visualize = Road2D(road_length, num_cars, lanes, vmax, p, traffic_lights=traffic_lights_visualize)

for step in range(steps):
    draw_road_state(road_visualize, step)
    road_visualize.update()


densities = np.linspace(3, 100, 10)  # плотность в машинах на км
ps = [0.0, 0.2, 0.5]
results = {}

# Сравнение с перестроением без светофора при разных вероятностях случайного возмущения
for p in ps:
    avg_speeds = []
    for density in densities:
        num_cars = int(density * road_length * cell_length / 1000)
        road_p = Road2D(road_length, num_cars, lanes, vmax, p)
        total_speed = 0
        for _ in range(steps):
            road_p.update()
            total_speed += road_p.get_average_speed_kmh()
        avg_speeds.append(total_speed / steps)
    results[p] = avg_speeds

plot_average_speed_density(densities, results)"""
