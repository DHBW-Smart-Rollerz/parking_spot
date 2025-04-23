import itertools
import math
import numpy as np
from itertools import combinations, permutations


# standard config
config = {
    "short_side": 380,
    "long_side": 490,
    "length_tolerance": 0.15,
    "angle_tolerance": 10,
    "vertical_tolerance": 10,
}


def is_right_angle(v1, v2, angle_tolerance):
    angle = abs(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return abs(angle) <= angle_tolerance


def get_angle(v1, v2):
    return abs(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def angle_between_three_points(p1, p2, p3):
    p1 = np.array(p1).reshape(-1)
    p2 = np.array(p2).reshape(-1)
    p3 = np.array(p3).reshape(-1)
    a = p1 - p2
    b = p3 - p2
    cos_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    angle_rad = math.acos(np.clip(cos_angle, -1.0, 1.0))
    return math.degrees(angle_rad)


def distance(p1, p2):
    return np.linalg.norm(np.array(p1).reshape(-1) - np.array(p2).reshape(-1))


def angle_to_x_axis(p1, p2):
    
    try:
        p_1 = p1[0]
        p_2 = p2[0]
        dx = p_2[0] - p_1[0]
        dy = p_2[1] - p_1[1]
    except:
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    return angle_deg % 360


def normalize_point(p):
    p = np.array(p).flatten()
    if len(p) >= 2:
        return (p[0], p[1])
    raise ValueError(f"Ungültiger Punkt: {p}")


def find_collinear_triplets(
    points,
    angle_threshold=config["angle_tolerance"],
    min_distance=config["short_side"] * (1 - config["length_tolerance"]),
    max_distance=config["short_side"] * (1 + config["length_tolerance"]),
    vertical_angle_tolerance=config["vertical_tolerance"],
):
    """
    Findet alle Tripel oder Vierergruppen, bei denen die Punkte nahezu auf einer Linie liegen,
    die benachbarten Punkte innerhalb einer Distanzspanne liegen und die Gesamtlinie nahezu
    senkrecht zur x-Achse steht.
    """
    collinear_groups = []

    if len(points) > 2:
        for trip in itertools.combinations(points, 3):
            sorted_trip = sorted(trip, key=lambda p: (p[0][0], p[0][1]))
            p1, p2, p3 = sorted_trip
            d12 = distance(p1, p2)
            d23 = distance(p2, p3)
            d13 = distance(p1, p3)
            if (
                min_distance <= d12 <= max_distance
                and min_distance <= d23 <= max_distance
                and 2 * min_distance <= d13 <= max_distance * 2
            ):
                angle = angle_between_three_points(p1, p2, p3)
                if 180 - angle_threshold <= angle <= 180 + angle_threshold:
                    vertical_angle = angle_to_x_axis(p1, p3)
                    if (
                        abs(vertical_angle - 180) <= vertical_angle_tolerance
                        or abs(vertical_angle) <= vertical_angle_tolerance
                    ):
                        collinear_groups.append((p1))
                        collinear_groups.append((p2))
                        collinear_groups.append((p3))

    print("Anzahl der gefundenen Tripel:", len(collinear_groups))
    print(collinear_groups)
    unique_list = filter_doubles(collinear_groups)
    return unique_list

def filter_doubles(points):
    seen = set()
    new_points = []
    for point in points:
        key = tuple(point[0])  
        if key not in seen:
            seen.add(key)
            new_points.append(point)
    sorted_points = sorted(new_points, key=lambda point: point[0][0])
    return sorted_points


def calc_backsite_points(points, long_side=config["long_side"]):
    try:
        x1, y1, z1 = points[0]
        x2, y2, z1 = points[1]
    except:
        x1, y1, z1 = points[0][0]
        x2, y2, z1 = points[1][0]

    dx = x2 - x1
    dy = y2 - y1

    small_side = math.hypot(dx, dy)
    ndx = (dy / small_side) * long_side
    ndy = (-dx / small_side) * long_side

    a = np.array([x1, y1, 0])  
    b = np.array([x2, y2, 0])
    c = np.array([x1 - ndx, y1 - ndy, 0])
    d = np.array([x2 - ndx, y2 - ndy, 0])
    center = np.array([(a[0] + b[0] + c[0] + d[0]) / 4, (a[1] + b[1] + c[1] + d[1]) / 4, 0])

    return np.array([a, b, c, d, center])

def calc_route(x_p, y_p, R, entry_length=100, straight_step=80, curve_step=15):
    route = []


    # Gerade auf der Straße
    x = 0
    while x < x_p*0.59:
        route.append((x, 0, 0))
        x += straight_step

    return route

def generate_relative_points(anchor):
    relative_offsets = {
        "J": (-0.02 * anchor[0], -0.59 * anchor[1]),
        "I": (-0.03 * anchor[0], -0.65 * anchor[1]),
        "H": (-0.05 * anchor[0], -0.71 * anchor[1]),
        "G": (-0.06 * anchor[0], -0.77 * anchor[1]),
        "F": (-0.08 * anchor[0], -0.83 * anchor[1]),
        "E": (-0.11 * anchor[0], -0.89 * anchor[1]),
        "D": (-0.14 * anchor[0], -0.94 * anchor[1]),
        "C": (-0.24 * anchor[0], -0.98 * anchor[1]),
        "B": (-0.31 * anchor[0], -1.00 * anchor[1]),
        "A": (-1 * anchor[0], -1 * anchor[1]),
    }
    
    ret = [(anchor[0] + dx, anchor[1] + dy, 0) for name, (dx, dy) in relative_offsets.items()]
    return ret