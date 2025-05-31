import numpy as np
import random

class AntColony:
    def __init__(self, distances, n_ants, n_iterations, decay=0.1, alpha=1, beta=1):
        self.distances = distances
        self.pheromone = np.ones(distances.shape) / len(distances)
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
    
    def run(self):
        best_path = None
        best_distance = float('inf')
        
        for _ in range(self.n_iterations):
            paths = self._gen_paths()
            self._update_pheromone(paths)
            
            shortest_path = min(paths, key=lambda x: x[1])
            if shortest_path[1] < best_distance:
                best_path = shortest_path[0]
                best_distance = shortest_path[1]
        
        return best_path, best_distance
    
    def _gen_paths(self):
        paths = []
        for _ in range(self.n_ants):
            path = self._gen_path(0)  # начинаем с города 0
            distance = self._calc_distance(path)
            paths.append((path, distance))
        return paths
    
    def _gen_path(self, start):
        path = [start]
        visited = set(path)
        
        while len(visited) < len(self.distances):
            next_city = self._select_next(path[-1], visited)
            path.append(next_city)
            visited.add(next_city)
        
        return path
    
    def _select_next(self, current, visited):
        pheromone = self.pheromone[current] ** self.alpha
        heuristic = (1 / (self.distances[current] + 1e-10)) ** self.beta
        
        probs = pheromone * heuristic
        probs[list(visited)] = 0
        probs /= probs.sum()
        
        return np.random.choice(range(len(self.distances)), p=probs)
    
    def _calc_distance(self, path):
        return sum(self.distances[path[i]][path[i+1]] for i in range(len(path)-1)) + self.distances[path[-1]][path[0]]
    
    def _update_pheromone(self, paths):
        self.pheromone *= (1 - self.decay)
        
        for path, dist in paths:
            for i in range(len(path)-1):
                self.pheromone[path[i]][path[i+1]] += 1 / dist
            self.pheromone[path[-1]][path[0]] += 1 / dist