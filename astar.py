import pygame  
import math
from queue import PriorityQueue, Queue, LifoQueue

WIDTH = 800
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("Path Finding Algorithms")

# Definición de colores
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)

class Spot:
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.color = WHITE
        self.neighbors = []
        self.width = width
        self.total_rows = total_rows
        self.cost = float("inf")  
        self.f_score = float("inf")

    def __lt__(self, other):
        return self.cost < other.cost

    def get_pos(self):
        return self.row, self.col

    def is_closed(self):
        return self.color == RED

    def is_open(self):
        return self.color == GREEN

    def is_barrier(self):
        return self.color == BLACK

    def is_start(self):
        return self.color == ORANGE

    def is_end(self):
        return self.color == TURQUOISE

    def reset(self):
        self.color = WHITE

    def make_start(self):
        self.color = ORANGE

    def make_closed(self):
        self.color = RED

    def make_open(self):
        self.color = GREEN

    def make_barrier(self):
        self.color = BLACK

    def make_end(self):
        self.color = TURQUOISE

    def make_path(self):
        self.color = PURPLE

    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

    def update_neighbors(self, grid):
        self.neighbors = []
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier():  # DOWN
            self.neighbors.append(grid[self.row + 1][self.col])
        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier():  # UP
            self.neighbors.append(grid[self.row - 1][self.col])
        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier():  # RIGHT
            self.neighbors.append(grid[self.row][self.col + 1])
        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier():  # LEFT
            self.neighbors.append(grid[self.row][self.col - 1])

def h(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)

def print_path(came_from, current):
    path = []
    while current in came_from:
        path.append(current.get_pos())
        current = came_from[current]
    path = path[::-1]
    print("Path")
    print("Start:")
    for coord in path:
        print(f"  {coord},")
    print("Goal")
    print(f"Length: {len(path)} steps")

def reconstruct_path(came_from, current, draw):
    while current in came_from:
        current = came_from[current]
        current.make_path()
        draw()

def make_grid(rows, width):
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            spot = Spot(i, j, gap, rows)
            grid[i].append(spot)
    return grid

def draw_grid(win, rows, width):
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
    for j in range(rows):
        pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))

def draw(win, grid, rows, width):
    win.fill(WHITE)
    for row in grid:
        for spot in row:
            spot.draw(win)
    draw_grid(win, rows, width)
    pygame.display.update()

def get_clicked_pos(pos, rows, width):
    gap = width // rows
    y, x = pos
    row = y // gap
    col = x // gap
    return row, col

# Depth First Search
def dfs(draw, grid, start, end):
    stack = LifoQueue()
    stack.put(start)
    visited = {start}
    path_map = {}

    while not stack.empty():
        current_spot = stack.get()

        if current_spot == end:
            reconstruct_path(path_map, end, draw)
            print_path(path_map, end)
            end.make_end()
            return True

        for adjacent in current_spot.neighbors:
            if adjacent not in visited:
                visited.add(adjacent)
                path_map[adjacent] = current_spot
                stack.put(adjacent)
                adjacent.make_open()

        draw()
        if current_spot != start:
            current_spot.make_closed()

    return False

# Breadth First Search
def bfs(draw, grid, start, end):
    search_queue = Queue()
    search_queue.put(start)
    path_map = {}
    explored = {start}

    while not search_queue.empty():
        current_spot = search_queue.get()
        
        if current_spot == end:
            reconstruct_path(path_map, end, draw)
            print_path(path_map, end)
            end.make_end()
            return True

        for adjacent in current_spot.neighbors:
            if adjacent not in explored:
                path_map[adjacent] = current_spot
                explored.add(adjacent)
                search_queue.put(adjacent)
                adjacent.make_open()

        draw()
        if current_spot != start:
            current_spot.make_closed()
            
    return False

def ucs(draw, grid, start, end):
    priority_queue = PriorityQueue()
    priority_queue.put((0, start))
    came_from_dict = {}
    cost_from_start = {spot: float("inf") for row in grid for spot in row}
    cost_from_start[start] = 0

    while not priority_queue.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current_cost, current_node = priority_queue.get()

        if current_node == end:
            reconstruct_path(came_from_dict, end, draw)
            print_path(came_from_dict, end)
            end.make_end()
            return True

        for neighbor in current_node.neighbors:
            new_cost = current_cost + 1
            if new_cost < cost_from_start[neighbor]:
                came_from_dict[neighbor] = current_node
                cost_from_start[neighbor] = new_cost
                neighbor.f_score = new_cost  # Actualiza el f_score del vecino
                priority_queue.put((new_cost, neighbor))
                neighbor.make_open()

        draw()
        if current_node != start:
            current_node.make_closed()

    return False

# Dijkstra
def dijkstra(draw, grid, start, end):
    return ucs(draw, grid, start, end)  # Specific case of UCS where all costs are equal

# A*
def a_star_algorithm(draw, grid, start, end):
    node_count = 0
    priority_queue = PriorityQueue()
    priority_queue.put((0, node_count, start))
    came_from_dict = {}
    cost_from_start = {spot: float("inf") for row in grid for spot in row}
    cost_from_start[start] = 0
    estimated_total_cost = {spot: float("inf") for row in grid for spot in row}
    estimated_total_cost[start] = h(start.get_pos(), end.get_pos())
    open_set_nodes = {start}

    while not priority_queue.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current_node = priority_queue.get()[2]
        open_set_nodes.remove(current_node)

        if current_node == end:
            reconstruct_path(came_from_dict, end, draw)
            print_path(came_from_dict, end)
            end.make_end()
            return True

        for neighbor in current_node.neighbors:
            temp_g_score = cost_from_start[current_node] + 1

            if temp_g_score < cost_from_start[neighbor]:
                came_from_dict[neighbor] = current_node
                cost_from_start[neighbor] = temp_g_score
                estimated_total_cost[neighbor] = temp_g_score + h(neighbor.get_pos(), end.get_pos())
                if neighbor not in open_set_nodes:
                    node_count += 1
                    priority_queue.put((estimated_total_cost[neighbor], node_count, neighbor))
                    open_set_nodes.add(neighbor)
                    neighbor.make_open()

        draw()
        if current_node != start:
            current_node.make_closed()

    return False

def main(win, width):
    ROWS = 50
    grid = make_grid(ROWS, width)

    start = None
    end = None
    run = True
    while run:
        draw(win, grid, ROWS, width)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if pygame.mouse.get_pressed()[0]:  
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                spot = grid[row][col]
                if not start and spot != end:
                    start = spot
                    start.make_start()
                elif not end and spot != start:
                    end = spot
                    end.make_end()
                elif spot != end and spot != start:
                    spot.make_barrier()

            elif pygame.mouse.get_pressed()[2]:  
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                spot = grid[row][col]
                spot.reset()
                if spot == start:
                    start = None
                elif spot == end:
                    end = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_d and start and end:
                    for row in grid:
                        for spot in row:
                            spot.update_neighbors(grid)
                    dfs(lambda: draw(win, grid, ROWS, width), grid, start, end)
                if event.key == pygame.K_a and start and end:
                    for row in grid:
                        for spot in row:
                            spot.update_neighbors(grid)
                    a_star_algorithm(lambda: draw(win, grid, ROWS, width), grid, start, end)
				if event.key == pygame.K_b and start and end:
                    for row in grid:
                        for spot in row:
                            spot.update_neighbors(grid)
                    bfs(lambda: draw(win, grid, ROWS, width), grid, start, end)
                if event.key == pygame.K_u and start and end:
                    for row in grid:
                        for spot in row:
                            spot.update_neighbors(grid)
                    ucs(lambda: draw(win, grid, ROWS, width), grid, start, end)
                if event.key == pygame.K_k and start and end:
                    for row in grid:
                        for spot in row:
                            spot.update_neighbors(grid)
                    dijkstra(lambda: draw(win, grid, ROWS, width), grid, start, end)
                if event.key == pygame.K_c:
                    start = None
                    end = None
                    grid = make_grid(ROWS, width)

    pygame.quit()

main(WIN, WIDTH)
