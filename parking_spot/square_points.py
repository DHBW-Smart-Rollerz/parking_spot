import itertools
import math
import numpy as np
from itertools import combinations, permutations


# standard config
config = {
    "short_side": 380,
    "long_side": 390,
    "length_tolerance": 0.15,
    "angle_tolerance": 5,
    "vertical_tolerance": 10,
}


def is_right_angle(v1, v2, angle_tolerance):
    angle = abs(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return abs(angle) <= angle_tolerance


def get_angle(v1, v2):
    return abs(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def find_rectangle(
    points,
    short_side=config["short_side"],
    long_side=config["long_side"],
    length_tolerance=config["length_tolerance"],
    angle_tolerance=config["angle_tolerance"],
):
    points = list(map(tuple, points))
    for quad in combinations(points, 4):
        for perm in permutations(quad):
            p1, p2, p3, p4 = perm

            d1, d2, d3, d4 = (
                distance(p1, p2),
                distance(p2, p3),
                distance(p3, p4),
                distance(p4, p1),
            )
            if all(
                any(
                    abs(d - target) <= length_tolerance
                    for target in [short_side, long_side]
                )
                for d in [d1, d2, d3, d4]
            ):
                v1 = np.array(p2) - np.array(p1)
                v2 = np.array(p3) - np.array(p2)
                v3 = np.array(p4) - np.array(p3)
                v4 = np.array(p1) - np.array(p4)

                if is_right_angle(v1, v2, angle_tolerance) and is_right_angle(
                    v2, v3, angle_tolerance
                ):
                    print(f"Gefundenes Rechteck: {p1}, {p2}, {p3}, {p4}")
                    print(f"Seitenlängen: {d1:.2f}, {d2:.2f}, {d3:.2f}, {d4:.2f}")
                    angle1 = np.degrees(
                        np.arccos(
                            np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        )
                    )
                    angle2 = np.degrees(
                        np.arccos(
                            np.dot(v2, v3) / (np.linalg.norm(v2) * np.linalg.norm(v3))
                        )
                    )
                    print(f"Winkel: {angle1:.2f}°, {angle2:.2f}°")
                    return [p1, p2, p3, p4]

    return None


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
    p1 = p1[0]
    p2 = p2[0]
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
                        abs(vertical_angle - 180) <= vertical_angle_tolerance
                        or abs(vertical_angle - 360) <= vertical_angle_tolerance
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
