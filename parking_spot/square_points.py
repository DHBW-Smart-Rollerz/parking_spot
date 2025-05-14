import itertools
import math
import numpy as np
from itertools import combinations, permutations
from parking_spot.config import PARKING_SPOT_CONFIG


# standard config
config = PARKING_SPOT_CONFIG


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


def generate_route_with_quarter_turn(xp, yp, r=5, num_points_curve=20):
    route = []

        
    for i in range(10):
        x = i * (xp - r) / 10  
        route.append((x, 0, 0))
        
    for i in range(num_points_curve + 1):
        # Berechnung des Winkels von 270° bis 360°
        theta = (math.pi / 2) * (i / num_points_curve) + (3 * math.pi / 2)  # Verschiebung um 270°
        
        # Berechnung der x- und y-Koordinaten
        x = xp - r + r * math.cos(theta)
        y = r + r * math.sin(theta)
        
        route.append((x, y, 0))
    
    for i in range(5):
        y = i * (yp - r) / 5  
        route.append((xp, y + r, 0))

    route_three_points = [(xp-r, 0, 0), (xp, yp-r, 0), (xp, yp, 0)]
    route.append((xp, yp, 0))
    route_sorted = sorted(route, key=lambda point: (point[0], point[1]))  # Zuerst nach x, dann nach y
    return route_sorted, route_three_points

def transform_points(xc, yc, yaw, local_points):
    """
        xc : current x-Position
        yc : current y-Position
        yaw : yaw in rad
        local_points (list of tuple): list of (xp, yp, zp) points of route relative to the vehicle

    Returns:
        list of points in world coordinates
    """
    R = np.array([
        [math.cos(yaw), -math.sin(yaw)],
        [math.sin(yaw),  math.cos(yaw)]
    ])
    
    world_points = []
    for xp, yp, zp in local_points:
        rel_vec = np.array([xp, yp])
        rotated = R @ rel_vec
        xw = xc + rotated[0]
        yw = yc + rotated[1]
        world_points.append((xw, yw))
    
    return world_points

def has_turned_90_degrees(old_angle_rad, new_angle_rad, tolerance_deg = config["turn_tolerance_in_deg"]):
    
    tolerance_rad = math.radians(tolerance_deg)
    angle_diff = (new_angle_rad - old_angle_rad + math.pi) % (2 * math.pi) - math.pi  

    return abs(abs(angle_diff) - math.pi/2) <= tolerance_rad