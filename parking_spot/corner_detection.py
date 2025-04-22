#!/usr/bin/env python3
from typing import Any
import abc

import time
import cv2 as cv
import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
from camera_preprocessing.transformation.coordinate_transform import (
    CoordinateTransform,
)
from parking_spot.square_points import *
from scipy.integrate import quad

config = {
    "short_side": 380,
    "long_side": 490,
    "length_tolerance": 0.15,
    "angle_tolerance": 10,
    "vertical_tolerance": 10,
    "interestx": 255,
    "interesty": 280,
}


class Detection(abc.ABC):
    """Abstracte Klasse für alle verschiedenen Erkennungsalgorithmen im Parkplatz-Szenario."""

    def __init__(self, log_level: Any = "ERROR") -> None:
        """Initialisiert die Erkennungsklasse.

        Args:
            log_level (Any, optional): Das Log-Level. Standard ist 'ERROR'.
        """
        self._log_level: Any = log_level
        self._last_result: Any = None

    def reset(self) -> None:
        """Setzt das letzte Ergebnis auf None zurück."""
        self._last_result = None

    def get_result(self) -> Any:
        """Gibt das letzte Ergebnis zurück.

        Returns:
            Any: Letztes Ergebnis.
        """
        return self._last_result

    @abc.abstractmethod
    def find(self, img: np.ndarray) -> None:
        """Findet das Merkmal im Bild und speichert das Ergebnis.

        Args:
            img (np.ndarray): Bild, in dem das Merkmal gesucht wird.
        """
        raise NotImplementedError(
            "Diese Methode muss in der Kindklasse implementiert werden."
        )

    @abc.abstractmethod
    def draw(
        self, img: np.ndarray, color: list = [255, 0, 0], thickness: int = 3
    ) -> np.ndarray:
        """Zeichnet das Ergebnis in das übergebene Bild.

        Args:
            img (np.ndarray): Bild, in das gezeichnet werden soll.
            color (list, optional): Farbe der Markierung. Standard: [255, 0, 0]
            thickness (int, optional): Dicke der Markierung. Standard: 3.

        Returns:
            np.ndarray: Bild mit eingezeichnetem Merkmal.
        """
        raise NotImplementedError(
            "Diese Methode muss in der Kindklasse implementiert werden."
        )

    def _log(self, msg: str, level: str = "INFO") -> None:
        """Loggt die Nachricht mit dem angegebenen Level.

        Args:
            msg (str): Nachricht zum Loggen.
            level (str, optional): Log-Level (INFO, DEBUG, WARN, ERROR, FATAL).
            Standard: 'INFO'.
        """
        levels = ["DEBUG", "INFO", "WARN", "ERROR", "FATAL"]
        if level not in levels:
            print(f"[ERROR] Unbekanntes Log-Level: {level}")
            print(f"[ERROR] {msg}")
        else:
            print(f"[{level}] {msg}")


class CornerDetection(Detection):
    """Klasse zur Eckenerkennung mit dem Harris Corner Detection Algorithmus."""

    BLOCK_SIZE = 11
    K_SIZE = 9
    K = 0.04

    def __init__(
        self,
        block_size: int = BLOCK_SIZE,
        k_size: int = K_SIZE,
        k: float = K,
        log_level: Any = "ERROR",
    ) -> None:
        super().__init__(log_level)
        self._block_size = block_size
        self._k_size = k_size
        self._k = k
        self._last_result = None
        self.transform = CoordinateTransform()

    def find(self, img: np.ndarray) -> None:
        """finds corners on baseline in area of interest
        points that are in a line and have parking spot distance

        Args:
            img (np.ndarray): gray scale img.

        Raises:
            ValueError: If the image is not a grayscale image.
        """
        if len(img.shape) != 2:
            raise ValueError("Das Bild muss ein Graustufenbild sein.")
        start = time.time()

        # Harris Corner Detection
        operatedImage = np.float32(img)

        dst = cv.cornerHarris(operatedImage, self.BLOCK_SIZE, self.K_SIZE, self._k)
        dst = cv.dilate(dst, None)

        # Schwellenwert zur Bestimmung der Ecken
        mask = np.zeros_like(img)
        mask[dst > 0.05 * dst.max()] = 255

        all_points = np.argwhere(mask)
        x = all_points[:, 1]
        y = all_points[:, 0]
        all_points = np.array([x, y]).T

        self._last_result = all_points
        ret = []
        # area of interest 0,0 to 250,280
        for corner in self._last_result:
            if corner[0] < config["interestx"] and corner[1] < config["interesty"]:
                ret.append((corner[0], corner[1]))
        self._last_result = ret
        data = np.array(ret, dtype=np.float32)
        eps = 2

        # Führe das Clustering durch
        if data.__len__():
            db = DBSCAN(eps=eps, min_samples=1).fit(data)
            labels = db.labels_

            # Berechne den Mittelwert für jede Clustergruppe
            unique_labels = set(labels)
            fused_points = []
            for label in unique_labels:
                cluster_points = data[labels == label]
                mean_point = np.mean(cluster_points, axis=0)
                fused_points.append(tuple(mean_point))
            self._last_result = fused_points
        # self._last_result = CornerDetection.merge_close_points(self._last_result, 6)

        finished = time.time()
        self._log(f"Zeit zur Eckenerkennung: {finished - start:.4f} Sekunden", "DEBUG")

        RT = []
        for i in self._last_result:
            RT.append(self.transform.camera_to_world(i))
        c = find_collinear_triplets(RT)
        self._last_result = []
        self._last_result_w = []
        self.parking_spots = []
        for i in c:
            self._last_result_w.append(i)
            self._last_result.append(self.transform.world_to_camera(i))
        return self._last_result_w

    def draw(
        self, img: np.ndarray, color=[0, 255, 0], thickness: int = 4,
    points = None) -> np.ndarray:
        """draws points on image
        """
        if points is None:
            points = self._last_result
        if points is None:
            return img

        for corner in points:
            x, y = corner.flatten()  # oder corner.ravel()
            img = cv.circle(img, (int(x), int(y)), thickness, [255, 0, 0], -1)
        self._log(f"{points.__len__()} Ecken erkannt")

        return img
    
    def getFullParkingSpots(self, points):
        """
        calculates backcorners by input of front corners
        returns list of tuples with all four corners
        """
        self.parking_spots = []
        self.parking_spots_w = []
        valid_spots = []

        # Versuche jede Kombination von benachbarten Punkten
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dist = distance(points[i], points[j])
                if not (config["short_side"] * (1 - config["length_tolerance"]) <= dist <= config["short_side"] * (1 + config["length_tolerance"])):
                    self._log(f"distance {dist} is not in range", "DEBUG")
                    continue

                spot = [points[i], points[j]]
                full_spot = calc_backsite_points(spot)

                try:
                    valid_spots.append(full_spot)
                except IndexError:
                    valid_spots.append([full_spot])
        

        unique_spots = remove_duplicates(valid_spots)
        self._log(f"unique_spots {unique_spots}", "DEBUG")

        for i in range(0, len(unique_spots)):
            for point in unique_spots[i]:
                try:
                    self.parking_spots[i].append(self.transform.world_to_camera(point))
                    self.parking_spots_w[i].append(point)
                except IndexError:
                    self.parking_spots.append([self.transform.world_to_camera(point)])
                    self.parking_spots_w.append([point])
        return self.parking_spots, self.parking_spots_w

    def draw_spots(
        self, img: np.ndarray, color=[0, 255, 0], thickness: int = 4, 
        spots = None) -> np.ndarray:
        """draws points on image
        """
        if spots is None:
            return img
        self._log(f"Anzahl der Parkplätze: {spots.__len__()}", "DEBUG")

        for spot in spots:
            points = spot[:4]
            pts = np.array([p[0] for p in points])

            # nach y sortieren: oben zuerst
            sorted_by_y = pts[np.argsort(pts[:, 1])]

            top = sorted_by_y[:2]
            bottom = sorted_by_y[2:4]

            # innerhalb top/bottom nach x sortieren
            top_left, top_right = top[np.argsort(top[:, 0])]
            bottom_left, bottom_right = bottom[np.argsort(bottom[:, 0])]
            pts = np.array([top_right, bottom_right, bottom_left, top_left])
            for i in range(0, 4):
                if (i<3):
                    x, y = pts[i].flatten()
                    x2, y2 = pts[i+1].flatten()
                else:
                    x, y = pts[i].flatten()
                    x2, y2 = pts[0].flatten()
                img = cv.line(img, (int(x), int(y)), (int(x2), int(y2)), (0, 0, 255), 2)

            x, y = spot[4].flatten()
            if ((0 <= x <= 800) and (0 <= y <= 640)):
                img = cv.circle(img, (int(x), int(y)), thickness, [0, 255, 0], -1)

        return img
    

    # Generierung der Route entlang der Clothoide
    def generate_clothoide_route(x_origin, y_origin, r, a, num_points=100):

        route = []

        # Bogenlängen-Werte für die Route
        s_values = np.linspace(0, 10 * r, num_points)  # Strecke entlang der Clothoide

        for s in s_values:
            x = x_origin + clothoide_x(s, a)
            y = y_origin + clothoide_y(s, a)
            route.append(x, y, 0)
            

def remove_duplicates(spots):
    unique = []
    for spot in spots:
        if not any(are_same_spot(spot, u) for u in unique):
            unique.append(spot)
    return unique
    
    
def are_same_spot(p1, p2):
    p1 = p1[0]  
    p2 = p2[0]  

    return (
        (np.allclose(p1[0], p2[0]) and np.allclose(p1[1], p2[1])) or
        (np.allclose(p1[0], p2[1]) and np.allclose(p1[1], p2[0]))
    )

def clothoide_x(s, a):
    return quad(lambda tau: np.cos((a**2 / 2) * tau**2), 0, s)[0]

def clothoide_y(s, a):
    return quad(lambda tau: np.sin((a**2 / 2) * tau**2), 0, s)[0]