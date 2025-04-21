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


def generate_parking_route_world(start, a, b, c, d, turn_radius_m=3.0, steps_per_meter=10):
    """
    Generiert eine Route vom Startpunkt in den Parkplatz a-b-c-d in Weltkoordinaten.
    
    Parameter:
    - start: Startposition [x, y]
    - a, b: Vorderseite des Parkplatzes (an der Straße)
    - c, d: Rückseite
    - turn_radius_m: Wendekreisradius in Metern
    - steps_per_meter: wie viele Punkte pro Meter auf dem Bogen erzeugt werden

    Rückgabe:
    - Liste von [x, y] Punkten, die die Route beschreiben
    """
    start = np.array(start[:2])
    a, b, c, d = map(lambda p: np.array(p[:2]), (a, b, c, d))

    # Mittelpunkt der Parkplatzfront (Einfahrt)
    entry = (a + b) / 2
    front_vec = b - a
    front_vec = front_vec / np.linalg.norm(front_vec)

    # Normalenvektor zur Parkplatzfront (zeigt auf Straße)
    normal = np.array([-front_vec[1], front_vec[0]])

    # Punkt vor dem Parkplatz entlang der Einfahrt
    pre_entry = entry - normal * turn_radius_m

    # Kreisbahn vorbereiten: Wir wollen von rechts kommend (auf rechter Spur) in Richtung pre_entry biegen
    dir_vec = (pre_entry - start)
    dir_vec = dir_vec / np.linalg.norm(dir_vec)
    ortho = np.array([-dir_vec[1], dir_vec[0]])

    # Mittelpunkt des Wendebogens
    circle_center = start + ortho * turn_radius_m

    # Start- und Endwinkel berechnen
    start_angle = np.arctan2(start[1] - circle_center[1], start[0] - circle_center[0])
    end_angle = np.arctan2(pre_entry[1] - circle_center[1], pre_entry[0] - circle_center[0])

    clockwise = np.cross(dir_vec, pre_entry - start) < 0
    if clockwise and end_angle > start_angle:
        end_angle -= 2 * np.pi
    elif not clockwise and end_angle < start_angle:
        end_angle += 2 * np.pi

    # Kreisbogenpunkte
    arc_len = abs(end_angle - start_angle) * turn_radius_m
    num_points = max(2, int(arc_len * steps_per_meter))
    angles = np.linspace(start_angle, end_angle, num_points)

    arc_points = [
        [
            circle_center[0] + turn_radius_m * np.cos(a),
            circle_center[1] + turn_radius_m * np.sin(a)
        ] for a in angles
    ]

    # Gerade Linie: pre_entry → entry → Parkplatzmitte
    center = (c + d) / 2
    line_points = [pre_entry.tolist(), entry.tolist(), center.tolist()]

    return [start.tolist()] + arc_points + line_points

