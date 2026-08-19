"""
Centroid-based object tracker for maintaining object identities across frames.
"""

import numpy as np


class CentroidTracker:
    """
    A simple centroid tracker that associates object centroids frame-to-frame
    using Euclidean distance and a maximum allowed distance threshold.

    Attributes:
        max_disappeared (int): Maximum number of frames an object can be
            missing before it is deregistered.
        max_distance (float): Maximum Euclidean distance between centroids
            to consider them the same object.
        next_object_id (int): Next available object ID.
        objects (dict[int, tuple[float, float]]): Mapping of object ID to
            its current centroid (x, y).
        disappeared (dict[int, int]): Mapping of object ID to the number of
            consecutive frames it has been missing.
    """

    def __init__(self, max_disappeared: int = 10, max_distance: float = 50.0):
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.next_object_id = 0
        self.objects: dict[int, tuple[float, float]] = {}
        self.disappeared: dict[int, int] = {}

    def update(self, centroids: list[tuple[float, float]]) -> dict[int, tuple[float, float]]:
        """
        Update the tracker with the latest set of centroids.

        Args:
            centroids: A list of (x, y) tuples representing detected object
                centroids in the current frame.

        Returns:
            The updated dictionary of tracked objects (ID -> centroid).
        """
        # If no centroids are provided, mark all existing objects as disappeared
        if len(centroids) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.objects.pop(obj_id, None)
                    self.disappeared.pop(obj_id, None)
            return self.objects

        # Convert input centroids to a numpy array for vectorized distance computation
        input_centroids = np.array(centroids, dtype=float)

        # If we have no tracked objects, register all input centroids
        if len(self.objects) == 0:
            for centroid in input_centroids:
                self._register(tuple(centroid))
        else:
            # Get existing object IDs and centroids
            object_ids = list(self.objects.keys())
            object_centroids = np.array([self.objects[oid] for oid in object_ids])

            # Compute Euclidean distance matrix (rows = existing, cols = new)
            distance_matrix = np.linalg.norm(
                object_centroids[:, np.newaxis] - input_centroids[np.newaxis, :],
                axis=2
            )

            # Sort by minimum distance and assign closest pairs
            sorted_rows = distance_matrix.min(axis=1).argsort()
            sorted_cols = distance_matrix.argmin(axis=1)[sorted_rows]

            used_rows = set()
            used_cols = set()

            for row, col in zip(sorted_rows, sorted_cols):
                if row in used_rows or col in used_cols:
                    continue
                if distance_matrix[row, col] > self.max_distance:
                    continue

                obj_id = object_ids[row]
                self.objects[obj_id] = tuple(input_centroids[col])
                self.disappeared[obj_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            # Identify rows (existing objects) that were not matched
            unused_rows = set(range(distance_matrix.shape[0])) - used_rows
            # Identify columns (new centroids) that were not matched
            unused_cols = set(range(distance_matrix.shape[1])) - used_cols

            # If there are more existing objects than new centroids,
            # mark the unmatched existing objects as disappeared
            if distance_matrix.shape[0] >= distance_matrix.shape[1]:
                for row in unused_rows:
                    obj_id = object_ids[row]
                    self.disappeared[obj_id] += 1
                    if self.disappeared[obj_id] > self.max_disappeared:
                        self.objects.pop(obj_id, None)
                        self.disappeared.pop(obj_id, None)
            else:
                # Otherwise, register each unmatched new centroid as a new object
                for col in unused_cols:
                    self._register(tuple(input_centroids[col]))

        return self.objects

    def _register(self, centroid: tuple[float, float]) -> None:
        """
        Register a new object with the next available ID.

        Args:
            centroid: The (x, y) centroid of the new object.
        """
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1