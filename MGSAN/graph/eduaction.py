# coding: utf-8
"""
Graph definitions for EduAction dataset with COCO-WholeBody keypoints
Supports multiple keypoint subsets for different body part configurations.

Keypoint layout (133 total):
- 0-16: Body (17 points)
- 17-22: Feet (6 points)
- 23-90: Face (68 points)
- 91-111: Left hand (21 points)
- 112-132: Right hand (21 points)
"""
import numpy as np


# Keypoint subset definitions
KEYPOINT_SUBSETS = {
    'full_body': list(range(133)),  # 133 keypoints
    'upper_body': list(range(0, 17)) + list(range(23, 133)),  # 127 keypoints
    'body_only': list(range(0, 23)),  # 23 keypoints
    'body_no_feet': list(range(0, 17)),  # 17 keypoints
    'face_hands': list(range(23, 133)),  # 110 keypoints
    'hands_only': list(range(91, 133)),  # 42 keypoints
    'body_hands': list(range(0, 23)) + list(range(91, 133)),  # 65 keypoints
    'upper_body_hands': list(range(0, 11)) + list(range(91, 133)),  # 53 keypoints
    'face_only': list(range(23, 91)),  # 68 keypoints
}


def get_full_body_edges():
    """Get all edge connections for full 133 keypoint skeleton."""
    edges = []

    # Body connections (COCO-17 format)
    body_edges = [
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
    edges.extend(body_edges)

    # Feet connections (17-22)
    feet_edges = [
        (15, 17), (15, 18), (15, 19),  # Left foot
        (16, 20), (16, 21), (16, 22),  # Right foot
        (17, 18), (17, 19), (18, 19),  # Left foot internal
        (20, 21), (20, 22), (21, 22),  # Right foot internal
    ]
    edges.extend(feet_edges)

    # Face connections (23-90, 68 points) - simplified
    # Face contour (23-39)
    for i in range(23, 39):
        edges.append((i, i + 1))
    edges.append((39, 23))

    # Left eyebrow (40-44)
    for i in range(40, 44):
        edges.append((i, i + 1))

    # Right eyebrow (45-49)
    for i in range(45, 49):
        edges.append((i, i + 1))

    # Nose (50-58)
    for i in range(50, 58):
        edges.append((i, i + 1))

    # Left eye (59-64)
    for i in range(59, 64):
        edges.append((i, i + 1))
    edges.append((64, 59))

    # Right eye (65-70)
    for i in range(65, 70):
        edges.append((i, i + 1))
    edges.append((70, 65))

    # Outer lips (71-82)
    for i in range(71, 82):
        edges.append((i, i + 1))
    edges.append((82, 71))

    # Inner lips (83-90)
    for i in range(83, 89):
        edges.append((i, i + 1))
    edges.append((89, 83))

    # Connect face to body
    edges.append((0, 50))  # Body nose to face nose

    # Left hand connections (91-111, 21 points)
    left_hand_edges = [
        (9, 91),  # Body wrist to hand wrist
        (91, 92), (92, 93), (93, 94), (94, 95),  # Thumb
        (91, 96), (96, 97), (97, 98), (98, 99),  # Index
        (91, 100), (100, 101), (101, 102), (102, 103),  # Middle
        (91, 104), (104, 105), (105, 106), (106, 107),  # Ring
        (91, 108), (108, 109), (109, 110), (110, 111),  # Pinky
    ]
    edges.extend(left_hand_edges)

    # Right hand connections (112-132, 21 points)
    right_hand_edges = [
        (10, 112),  # Body wrist to hand wrist
        (112, 113), (113, 114), (114, 115), (115, 116),  # Thumb
        (112, 117), (117, 118), (118, 119), (119, 120),  # Index
        (112, 121), (121, 122), (122, 123), (123, 124),  # Middle
        (112, 125), (125, 126), (126, 127), (127, 128),  # Ring
        (112, 129), (129, 130), (130, 131), (131, 132),  # Pinky
    ]
    edges.extend(right_hand_edges)

    return edges


class GraphEduAction:
    """
    Graph for EduAction skeleton data.
    Supports different keypoint subsets via keypoint_subset parameter.
    """

    def __init__(self, labeling_mode='spatial', keypoint_subset='full_body'):
        self.labeling_mode = labeling_mode
        self.keypoint_subset = keypoint_subset

        # Get selected keypoints
        if keypoint_subset in KEYPOINT_SUBSETS:
            self.selected_keypoints = KEYPOINT_SUBSETS[keypoint_subset]
        else:
            self.selected_keypoints = KEYPOINT_SUBSETS['full_body']

        self.num_node = len(self.selected_keypoints)

        # Create index mapping
        self.idx_map = {orig: new for new, orig in enumerate(self.selected_keypoints)}

        # Build edges
        self.edge = self._build_edges()
        self.center = self._get_center()
        self.A = self.get_adjacency_matrix()

    def _build_edges(self):
        """Build edge list for selected keypoints."""
        # Self connections
        edges = [(i, i) for i in range(self.num_node)]

        # Get full body edges and filter/remap
        full_edges = get_full_body_edges()

        for v1, v2 in full_edges:
            if v1 in self.idx_map and v2 in self.idx_map:
                edges.append((self.idx_map[v1], self.idx_map[v2]))

        return edges

    def _get_center(self):
        """Get center node index for the selected subset."""
        # Try to use nose (0) as center if available
        if 0 in self.idx_map:
            return self.idx_map[0]
        # Otherwise use first node
        return 0

    def get_adjacency_matrix(self):
        """Build adjacency matrix."""
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge:
            if i < self.num_node and j < self.num_node:
                A[i, j] = 1
                A[j, i] = 1  # undirected
        return A


# Convenience classes for specific subsets
class GraphEduActionFullBody(GraphEduAction):
    """Full body with 133 keypoints."""
    def __init__(self, labeling_mode='spatial'):
        super().__init__(labeling_mode, 'full_body')


class GraphEduActionUpperBody(GraphEduAction):
    """Upper body without feet - 127 keypoints."""
    def __init__(self, labeling_mode='spatial'):
        super().__init__(labeling_mode, 'upper_body')


class GraphEduActionBodyOnly(GraphEduAction):
    """Body + feet only - 23 keypoints."""
    def __init__(self, labeling_mode='spatial'):
        super().__init__(labeling_mode, 'body_only')


class GraphEduActionBodyNoFeet(GraphEduAction):
    """Body only without feet - 17 keypoints."""
    def __init__(self, labeling_mode='spatial'):
        super().__init__(labeling_mode, 'body_no_feet')


class GraphEduActionFaceHands(GraphEduAction):
    """Face + hands - 110 keypoints."""
    def __init__(self, labeling_mode='spatial'):
        super().__init__(labeling_mode, 'face_hands')


class GraphEduActionHandsOnly(GraphEduAction):
    """Both hands only - 42 keypoints."""
    def __init__(self, labeling_mode='spatial'):
        super().__init__(labeling_mode, 'hands_only')


class GraphEduActionBodyHands(GraphEduAction):
    """Body + hands - 65 keypoints."""
    def __init__(self, labeling_mode='spatial'):
        super().__init__(labeling_mode, 'body_hands')


class GraphEduActionUpperBodyHands(GraphEduAction):
    """Upper body + hands - 53 keypoints."""
    def __init__(self, labeling_mode='spatial'):
        super().__init__(labeling_mode, 'upper_body_hands')


class GraphEduActionFaceOnly(GraphEduAction):
    """Face only - 68 keypoints."""
    def __init__(self, labeling_mode='spatial'):
        super().__init__(labeling_mode, 'face_only')
