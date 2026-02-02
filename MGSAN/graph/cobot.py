# coding: utf-8
import numpy as np

class GraphCobot:
    """ Graph for Cobot skeleton data """
    def __init__(self, labeling_mode='spatial'):
        self.num_node = 48
        self_link = [(i, i) for i in range(self.num_node)]
        neighbor_link = [
            (42, 44), (44, 46), (43, 45), (45, 47), (42, 43),
            (0, 1), (1,2) ,(2,3), (3,4),
            (0, 5), (5,6), (6,7), (7,8),
            (9,10), (10,11), (11,12),
            (13,14), (14,15), (15,16),
            (0,17), (17,18), (18,19), (19,20),
            (21, 22), (22, 23), (23, 24), (24, 25),
            (21, 26), (26, 27), (27, 28), (28, 29),
            (30, 31), (31, 32), (32, 33),
            (34, 35), (35, 36), (36, 37),
            (21, 38), (38, 39), (39, 40), (40, 41),
            (46,0), (47,21)
        ]
        self.edge = self_link + neighbor_link
        self.center = 42
        self.A = self.get_adjacency_matrix()

    def get_adjacency_matrix(self):
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge:
            A[i, j] = 1
            A[j, i] = 1  # undirected
        return A
