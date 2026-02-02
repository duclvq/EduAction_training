# coding: utf-8
"""
Graph for EduAction dataset with COCO-WholeBody 133 keypoints
Keypoint layout:
- 0-16: Body (17 points)
- 17-22: Feet (6 points)
- 23-90: Face (68 points)
- 91-111: Left hand (21 points)
- 112-132: Right hand (21 points)
"""
import numpy as np


class GraphEduAction:
    """ Graph for EduAction skeleton data with 133 keypoints """

    def __init__(self, labeling_mode='spatial'):
        self.num_node = 133
        self.labeling_mode = labeling_mode

        # Self connections
        self_link = [(i, i) for i in range(self.num_node)]

        # Body connections (COCO-17 format)
        # 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
        # 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
        # 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
        # 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
        body_links = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Face
            (0, 5), (0, 6),  # Nose to shoulders
            (5, 7), (7, 9),  # Left arm
            (6, 8), (8, 10),  # Right arm
            (5, 11), (6, 12),  # Shoulders to hips
            (11, 12),  # Hip connection
            (11, 13), (13, 15),  # Left leg
            (12, 14), (14, 16),  # Right leg
            (5, 6),  # Shoulder connection
        ]

        # Feet connections (17-22)
        # 17: left_big_toe, 18: left_small_toe, 19: left_heel
        # 20: right_big_toe, 21: right_small_toe, 22: right_heel
        feet_links = [
            (15, 17), (15, 18), (15, 19),  # Left foot
            (16, 20), (16, 21), (16, 22),  # Right foot
            (17, 18), (17, 19), (18, 19),  # Left foot internal
            (20, 21), (20, 22), (21, 22),  # Right foot internal
        ]

        # Face connections (23-90, 68 points) - simplified contour
        # Face outline, eyebrows, eyes, nose, lips
        face_links = []
        # Face contour (23-39)
        for i in range(23, 39):
            face_links.append((i, i + 1))
        face_links.append((39, 23))  # Close contour

        # Left eyebrow (40-44)
        for i in range(40, 44):
            face_links.append((i, i + 1))

        # Right eyebrow (45-49)
        for i in range(45, 49):
            face_links.append((i, i + 1))

        # Nose (50-58)
        for i in range(50, 58):
            face_links.append((i, i + 1))

        # Left eye (59-64)
        for i in range(59, 64):
            face_links.append((i, i + 1))
        face_links.append((64, 59))  # Close left eye

        # Right eye (65-70)
        for i in range(65, 70):
            face_links.append((i, i + 1))
        face_links.append((70, 65))  # Close right eye

        # Outer lips (71-82)
        for i in range(71, 82):
            face_links.append((i, i + 1))
        face_links.append((82, 71))  # Close outer lips

        # Inner lips (83-90)
        for i in range(83, 89):
            face_links.append((i, i + 1))
        face_links.append((89, 83))  # Close inner lips

        # Connect face to body (nose to face center)
        face_links.append((0, 50))  # Body nose to face nose

        # Left hand connections (91-111, 21 points)
        # 91: wrist, 92-95: thumb, 96-99: index, 100-103: middle
        # 104-107: ring, 108-111: pinky
        left_hand_links = [
            (9, 91),  # Body wrist to hand wrist
            (91, 92), (92, 93), (93, 94), (94, 95),  # Thumb
            (91, 96), (96, 97), (97, 98), (98, 99),  # Index
            (91, 100), (100, 101), (101, 102), (102, 103),  # Middle
            (91, 104), (104, 105), (105, 106), (106, 107),  # Ring
            (91, 108), (108, 109), (109, 110), (110, 111),  # Pinky
        ]

        # Right hand connections (112-132, 21 points)
        # 112: wrist, 113-116: thumb, 117-120: index, 121-124: middle
        # 125-128: ring, 129-132: pinky
        right_hand_links = [
            (10, 112),  # Body wrist to hand wrist
            (112, 113), (113, 114), (114, 115), (115, 116),  # Thumb
            (112, 117), (117, 118), (118, 119), (119, 120),  # Index
            (112, 121), (121, 122), (122, 123), (123, 124),  # Middle
            (112, 125), (125, 126), (126, 127), (127, 128),  # Ring
            (112, 129), (129, 130), (130, 131), (131, 132),  # Pinky
        ]

        # Combine all links
        neighbor_link = body_links + feet_links + face_links + left_hand_links + right_hand_links

        self.edge = self_link + neighbor_link
        self.center = 0  # Nose as center
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode='spatial'):
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge:
            if i < self.num_node and j < self.num_node:
                A[i, j] = 1
                A[j, i] = 1  # undirected
        return A
