import numpy as np
from scipy.spatial import KDTree

class PotentialField:
    def __init__(self, goal, obstacles, k_att, k_rep):
        self.goal = goal
        self.obstacles = obstacles
        self.k_att = k_att
        self.k_rep = k_rep
        self.kd_tree = KDTree(obstacles)

    def compute_attractive_force(self, position):
        return -self.k_att * (position - self.goal)

    def compute_repulsive_force(self, position):
        _, idx = self.kd_tree.query(position)
        nearest_obstacle = self.obstacles[idx]
        return self.k_rep * (1.0 / np.linalg.norm(position - nearest_obstacle) - 1.0 / np.linalg.norm(self.goal - nearest_obstacle))

    def compute_total_force(self, position):
        return self.compute_attractive_force(position) + self.compute_repulsive_force(position)

    def gradient_descent(self, start, step_size, max_iters):
        path = [start]
        for _ in range(max_iters):
            position = path[-1]
            force = self.compute_total_force(position)
            next_position = position - step_size * force
            path.append(next_position)
            if np.linalg.norm(next_position - self.goal) < step_size:
                break
        return path

# 创建一个示例数据集
points = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])

# 构建KDTree
tree = KDTree(points)

# 指定查询点
x = -1
y = -1

# 指定搜索半径
r = 2.9

# 查询距离点(x, y)在r范围内的所有点的索引
indices = tree.query_ball_point([x, y], r)

print(indices)