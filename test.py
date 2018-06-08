import numpy as np


def angle_of_lines(line1, line2):
    point1, point2 = line1
    point3, point4 = line2

    v1 = np.array((point2[0] - point1[0], point2[1] - point2[1]))
    v2 = np.array((point4[0] - point3[0], point4[1] - point3[1]))
    _dot = np.dot(v1, v2)

    dist1 = np.linalg.norm(np.array(point1) - np.array(point2))
    dist2 = np.linalg.norm(np.array(point3) - np.array(point4))
    theta = np.arccos(_dot / (dist1 * dist2)) / np.pi * 180
    return theta


linea = ((10, 10), (20, 10))
lineb = ((10, 10), (0, -0))
print(angle_of_lines(linea, lineb))
