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
    "vertical_tolerance": 15,
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
        for p1, p2, p3 in itertools.combinations(points, 3):

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
                        abs(vertical_angle - 90) <= vertical_angle_tolerance
                        or abs(vertical_angle - 270) <= vertical_angle_tolerance
                    ):
                        collinear_groups.append((p1))
                        collinear_groups.append((p2))
                        collinear_groups.append((p3))

    if len(points) > 3:
        for quad in itertools.combinations(points, 4):
            sorted_quad = sorted(quad, key=lambda p: (p[0][0], p[0][1]))
            a, b, c, d = sorted_quad
            d_ab = distance(a, b)
            d_bc = distance(b, c)
            d_cd = distance(c, d)

            if (
                min_distance <= d_ab <= max_distance
                and min_distance <= d_bc <= max_distance
                and min_distance <= d_cd <= max_distance
            ):
                angle1 = angle_between_three_points(a, b, c)
                angle2 = angle_between_three_points(b, c, d)
                if (
                    180 - angle_threshold <= angle1 <= 180 + angle_threshold
                    and 180 - angle_threshold <= angle2 <= 180 + angle_threshold
                ):
                    collinear_groups.append(a)
                    collinear_groups.append(b)
                    collinear_groups.append(c)
                    collinear_groups.append(d)

    return collinear_groups


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

    # Punkt, ab dem die Gerade endet und die Kurve beginnt
    x_entry = x_p - entry_length  # Beginn der Einfahrt zum Parkplatz (gerade Phase)
    x_curve_start = x_entry - math.sqrt(max(R**2 - (y_p - 0)**2, 0))

    # Gerade auf der Straße
    x = 0
    while x < x_curve_start:
        route.append((x, 0, 0))
        x += straight_step

    # Mittelpunkt der Kurve
    mx = x_entry
    my = y_p - R

    # Startpunkt der Kurve
    sx = x_curve_start
    sy = 0

    # Start- und Endwinkel der Kurve
    dx0 = sx - mx
    dy0 = sy - my
    theta_start = math.atan2(dy0, dx0)

    ex = x_entry
    ey = y_p - entry_length * (y_p / math.sqrt(x_p**2 + y_p**2))  # leicht angepasst
    dx1 = ex - mx
    dy1 = ey - my
    theta_end = math.atan2(dy1, dx1)

    if theta_end < theta_start:
        theta_end += 2 * math.pi

    arc_length = R * (theta_end - theta_start)
    steps = max(2, int(arc_length / curve_step))

    for i in range(steps + 1):
        theta = theta_start + (theta_end - theta_start) * i / steps
        x = mx + R * math.cos(theta)
        y = my + R * math.sin(theta)
        route.append((x, y, 0))

    # Gerade in den Parkplatz rein
    steps_entry = max(1, int(entry_length / straight_step))
    dx = (x_p - ex) / steps_entry
    dy = (y_p - ey) / steps_entry

    for i in range(1, steps_entry + 1):
        route.append((ex + i * dx, ey + i * dy, 0))

    return route
