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

    def find(self, img: np.ndarray) -> None:
        """Ermittelt Ecken im Graustufenbild mittels Harris Corner Detection.

        Args:
            img (np.ndarray): Graustufenbild.

        Raises:
            ValueError: Falls das Bild nicht in Graustufen vorliegt.
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
            if corner[0] < 250 and corner[1] < 280:
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
        transform = CoordinateTransform()
        RT = []
        for i in self._last_result:
            RT.append(transform.camera_to_world(i))
        self._last_result_all = self._last_result
        self._log(RT, "DEBUG")
        self._log(self._last_result, "DEBUG")
        c = find_collinear_triplets(RT)
        self._last_result = []
        self._last_result_w = []
        for i in c:
            self._last_result_w.append(i)
            print(i)
            self._last_result.append(transform.world_to_camera(i))
        return ret

    def draw(
        self, img: np.ndarray, color=[0, 255, 0], thickness: int = 4,
    points = None) -> np.ndarray:
        """Zeichnet erkannte Ecken in das Bild.

        Args:
            img (np.ndarray): Bild, in das gezeichnet wird.
            color (list, optional): Farbe der Punkte. Standard: [0, 255, 0].
            thickness (int, optional): Radius der Punkte. Standard: 1.

        Returns:
            np.ndarray: Bild mit eingezeichneten Ecken.
        """
        if points is None:
            points = self._last_result
        if points is None:
            return img
        self._log(f"Anzahl der Ecken: {points.__len__()}", "DEBUG")

        for corner in points:
            print(corner)
            x, y = corner.flatten()  # oder corner.ravel()
            img = cv.circle(img, (int(x), int(y)), thickness, [255, 0, 0], -1)
            self._log(f"{points.__len__()} Ecken erkannt")
        self._log(f"Cam: {points}")
        self._log(f"Real: {self._last_result_w}")

        return img
