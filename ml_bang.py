"""BANG-Clustering with a binary grid directory.

This module implements the clustering method described by Schikuta and Erhart
(1997).  The implementation deliberately stops at the binary directory used by
the paper and by the public pyclustering implementation.  It does not implement
a database-style BANG file with buckets, overflow handling, or a separate
nested-region directory.

The code is original and uses only NumPy and Matplotlib.  pyclustering and its
source are used only as a reference for the public API, region numbering, and
the order in which levels/channels are split.
"""

from __future__ import annotations

import enum
import math
from collections import deque

import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_VOLUME_FLOOR = float(np.finfo(float).tiny)


class ClusterEncoding(enum.IntEnum):
    """Equivalent values to pyclustering.cluster.encoder.type_encoding."""

    CLUSTER_INDEX_LABELING = 0
    CLUSTER_INDEX_LIST_SEPARATION = 1
    CLUSTER_OBJECT_LIST_SEPARATION = 2


type_encoding = ClusterEncoding


def _validate_data(data):
    array = np.asarray(data, dtype=float)
    if array.ndim != 2:
        raise ValueError("data must be a two-dimensional array.")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("data must contain at least one sample and one feature.")
    if not np.all(np.isfinite(array)):
        raise ValueError("data must not contain NaN or infinite values.")
    with np.errstate(over="ignore"):
        extents = np.max(array, axis=0) - np.min(array, axis=0)
    if not np.all(np.isfinite(extents)):
        raise ValueError(
            "Feature ranges overflow float64; normalize or rescale the data."
        )
    return np.ascontiguousarray(array)


def _validate_levels(levels):
    if isinstance(levels, bool) or not isinstance(levels, (int, np.integer)):
        raise TypeError("levels must be an integer.")
    levels = int(levels)
    if levels < 1:
        raise ValueError("levels must be at least 1.")
    return levels


def _validate_threshold(value, name):
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number.")
    return value


def _data_corners(data):
    return np.max(data, axis=0), np.min(data, axis=0)


def _volume_from_corners(max_corner, min_corner, volume_floor):
    sides = np.asarray(max_corner, dtype=float) - np.asarray(min_corner, dtype=float)
    positive_sides = sides[sides > 0.0]
    if positive_sides.size == 0:
        return 1.0
    with np.errstate(over="ignore", under="ignore"):
        volume = float(np.prod(positive_sides))
    if not np.isfinite(volume):
        raise ValueError(
            "A block volume overflowed float64; normalize or rescale the data."
        )
    return max(volume, float(volume_floor))


def _neighbor_geometry(max_a, min_a, max_b, min_b, tolerance=1e-9):
    """Classify the contact between two axis-aligned boxes.

    Returns (is_neighbor, neighbor_degree).  ``neighbor_degree`` is the
    dimensionality of the contact area as defined in Section 2.3 of the paper:
    the number of dimensions in which the two boxes overlap with positive
    length.  It is in [0, k-1] for two disjoint boxes that touch.
    """
    scale = max(
        float(np.max(np.asarray(max_a) - np.asarray(min_a))),
        float(np.max(np.asarray(max_b) - np.asarray(min_b))),
        1.0,
    )
    atol = float(tolerance) * scale
    overlap = np.minimum(max_a, max_b) - np.maximum(min_a, min_b)
    if np.any(overlap < -atol):
        return False, None
    if np.all(overlap > atol):
        return False, None
    touching = np.any(np.abs(overlap) <= atol)
    if not touching:
        return False, None
    degree = int(np.count_nonzero(overlap > atol))
    return True, degree


class SpatialBlock:
    """Rectangular region in the k-dimensional value space.

    The constructor follows pyclustering's ``spatial_block`` order:
    ``(max_corner, min_corner)``.  The attributes ``upper`` and ``lower`` are
    aliases used internally for readability.

    The paper defines the volume as the product of all k extents.  For the
    degenerate case in which a feature has zero extent, this implementation
    follows pyclustering's practical convention: zero-extent dimensions are
    omitted from the product.  If every dimension has zero extent, volume is
    defined as 1.0 so that a finite density can be computed.  These are
    numerical conventions, not claims of the paper.
    """

    def __init__(self, max_corner, min_corner, volume_floor=DEFAULT_VOLUME_FLOOR):
        max_corner = np.asarray(max_corner, dtype=float)
        min_corner = np.asarray(min_corner, dtype=float)
        if max_corner.ndim != 1 or max_corner.shape != min_corner.shape:
            raise ValueError("Corners must be one-dimensional arrays of the same shape.")
        if np.any(max_corner < min_corner):
            raise ValueError("Every max-corner coordinate must be >= its min-corner value.")
        if volume_floor <= 0.0:
            raise ValueError("volume_floor must be strictly positive.")
        self.max_corner = max_corner.copy()
        self.min_corner = min_corner.copy()
        self.volume_floor = float(volume_floor)
        self._volume = _volume_from_corners(
            self.max_corner, self.min_corner, self.volume_floor
        )

    @property
    def upper(self):
        return self.max_corner

    @property
    def lower(self):
        return self.min_corner

    @property
    def dimension(self):
        return self.min_corner.size

    def get_corners(self):
        return self.max_corner.tolist(), self.min_corner.tolist()

    def get_volume(self):
        return self._volume

    def contains(self, points):
        points = np.asarray(points, dtype=float)
        if points.ndim == 1:
            points = points[np.newaxis, :]
        if points.shape[1] != self.dimension:
            raise ValueError("Point dimension does not match the block dimension.")
        return np.all(
            (points >= self.min_corner) & (points <= self.max_corner), axis=1
        )

    def __contains__(self, point):
        return bool(self.contains([point])[0])

    def split(self, dimension):
        if not isinstance(dimension, (int, np.integer)):
            raise TypeError("dimension must be an integer.")
        if dimension < 0 or dimension >= self.dimension:
            raise ValueError("dimension is out of range.")
        border = 0.5 * (self.min_corner[dimension] + self.max_corner[dimension])
        left_max = self.max_corner.copy()
        left_max[dimension] = border
        right_min = self.min_corner.copy()
        right_min[dimension] = border
        return (
            SpatialBlock(left_max, self.min_corner, self.volume_floor),
            SpatialBlock(self.max_corner, right_min, self.volume_floor),
        )

    def is_neighbor(self, block, tolerance=1e-9):
        if not isinstance(block, SpatialBlock):
            raise TypeError("block must be a SpatialBlock.")
        if self.dimension != block.dimension:
            raise ValueError("Both blocks must have the same dimension.")
        neighbor, _ = _neighbor_geometry(
            self.max_corner,
            self.min_corner,
            block.max_corner,
            block.min_corner,
            tolerance,
        )
        return neighbor

    def neighbor_degree(self, block, tolerance=1e-9):
        if not isinstance(block, SpatialBlock):
            raise TypeError("block must be a SpatialBlock.")
        if self.dimension != block.dimension:
            raise ValueError("Both blocks must have the same dimension.")
        _, degree = _neighbor_geometry(
            self.max_corner,
            self.min_corner,
            block.max_corner,
            block.min_corner,
            tolerance,
        )
        return degree

    def __repr__(self):
        return (
            f"SpatialBlock(max_corner={self.max_corner.tolist()}, "
            f"min_corner={self.min_corner.tolist()})"
        )


class BANGBlock:
    """A grid region/block and the sample indices covered by it.

    ``region`` and ``level`` follow the BANG key convention.  A child block has
    a new level and its region number is derived from its parent's region and
    level.  They are deliberately not heap node ids.
    """

    def __init__(
        self,
        data,
        region,
        level,
        space_block,
        cache_points=False,
        point_indices=None,
    ):
        self.data = data
        self.region = int(region)
        self.level = int(level)
        if self.region < 0 or self.level < 0:
            raise ValueError("region and level must be non-negative.")
        self.spatial_block = space_block
        self.cluster = None
        self.children = []
        self.parent = None
        self.creation_index = 0
        self.cache_points = bool(cache_points)
        if point_indices is None:
            self._point_indices = None
        else:
            self._point_indices = np.asarray(point_indices, dtype=int)
        self._density = None

    @property
    def point_indices(self):
        if self._point_indices is None:
            self._point_indices = np.flatnonzero(
                self.spatial_block.contains(self.data)
            ).astype(int)
        return self._point_indices

    @property
    def size(self):
        return int(self.point_indices.size)

    @property
    def population(self):
        return self.size

    @property
    def density(self):
        if self._density is None:
            self._density = self._calculate_density()
        return self._density

    def _calculate_density(self):
        population = self.size
        if population == 0:
            return 0.0
        volume = self.spatial_block.get_volume()
        density = float(population) / volume
        if not np.isfinite(density):
            raise ValueError(
                "Block density is not finite; normalize or rescale the data, "
                "or reduce the number of levels/features."
            )
        return density

    def get_region(self):
        return self.region

    def get_density(self):
        return self.density

    def get_cluster(self):
        return self.cluster

    def get_spatial_block(self):
        return self.spatial_block

    def get_points(self):
        return self.point_indices.tolist()

    def set_cluster(self, index):
        self.cluster = None if index is None else int(index)

    def is_neighbor(self, block, tolerance=1e-9):
        return self.spatial_block.is_neighbor(block.spatial_block, tolerance)

    def neighbor_degree(self, block, tolerance=1e-9):
        return self.spatial_block.neighbor_degree(block.spatial_block, tolerance)

    def points(self):
        return self.data[self.point_indices]

    def split(self, dimension):
        left_space, right_space = self.spatial_block.split(dimension)
        border = left_space.max_corner[dimension]
        coordinates = self.data[self.point_indices, dimension]
        left_mask = coordinates < border
        left_indices = self.point_indices[left_mask]
        right_indices = self.point_indices[~left_mask]
        right_region = self.region + 2 ** self.level
        left = BANGBlock(
            self.data,
            self.region,
            self.level + 1,
            left_space,
            self.cache_points,
            left_indices,
        )
        right = BANGBlock(
            self.data,
            right_region,
            self.level + 1,
            right_space,
            self.cache_points,
            right_indices,
        )
        left.parent = self
        right.parent = self
        self.children = [left, right]
        return left, right

    def __len__(self):
        return self.size

    def __repr__(self):
        return (
            f"BANGBlock(region={self.region}, level={self.level}, "
            f"size={self.size}, density={self.density:.6g})"
        )


class BANGDirectory:
    """Binary BANG directory.

    Parameters
    ----------
    data, levels:
        Input data and number of tree levels, including the root level.  A
        positive number of levels means at most ``levels - 1`` splits along a
        root-to-leaf path.
    observe:
        If True, store each built level for O(1) access.  If False, only the
        tree and the leaf list are stored; ``get_level`` reconstructs a level by
        traversing the tree.  Internal blocks are retained in both modes, so
        this is not a complete memory-saving mode.
    density_threshold, amount_threshold:
        pyclustering-compatible stopping thresholds.  ``density_threshold``
        also defines which blocks are eligible for clustering; a block at or
        below it is noise.  ``amount_threshold`` only stops directory growth;
        it is not used as a cluster-candidate filter in this implementation.
        Both are software extensions, not parameters of the 1997 paper.
    """

    def __init__(self, data, levels, **kwargs):
        data = _validate_data(data)
        levels = _validate_levels(levels)
        self.observe = bool(kwargs.pop("observe", True))
        self.density_threshold = _validate_threshold(
            kwargs.pop("density_threshold", 0.0), "density_threshold"
        )
        amount_threshold = kwargs.pop("amount_threshold", 0)
        if isinstance(amount_threshold, bool) or not isinstance(
            amount_threshold, (int, np.integer)
        ):
            raise TypeError("amount_threshold must be an integer.")
        self.amount_threshold = int(amount_threshold)
        if self.amount_threshold < 0:
            raise ValueError("amount_threshold must be non-negative.")
        self.data = data
        self.levels = levels
        self.max_corner, self.min_corner = _data_corners(self.data)
        self.constant_dimensions = np.flatnonzero(
            self.max_corner == self.min_corner
        ).astype(int)
        root_space = SpatialBlock(self.max_corner, self.min_corner)
        self.root = BANGBlock(
            self.data,
            0,
            0,
            root_space,
            True,
            np.arange(self.data.shape[0], dtype=int),
        )
        self.level_blocks = [] if not self.observe else [[self.root]]
        self.leaf_blocks = []
        self._size = 1
        self._height = 1
        self._creation_counter = 0
        self.root.creation_index = 0
        self._build()
        self.leaf_blocks.sort(
            key=lambda block: (block.density, block.creation_index)
        )
        self.partition_check = self._check_partition()

    def _must_stop(self, block):
        if block.size == 0:
            return True
        if block.size <= self.amount_threshold:
            return True
        return block.density <= self.density_threshold

    def _build(self):
        current = [self.root]
        for level in range(1, self.levels):
            next_level = []
            for block in current:
                if self._must_stop(block):
                    self.leaf_blocks.append(block)
                    continue
                split_dimension = level % self.data.shape[1]
                left, right = block.split(split_dimension)
                self._creation_counter += 1
                left.creation_index = self._creation_counter
                self._creation_counter += 1
                right.creation_index = self._creation_counter
                next_level.extend((left, right))
                self._size += 2
            if not next_level:
                current = []
                break
            self._height = level + 1
            if self.observe:
                self.level_blocks.append(next_level)
            current = next_level
        self.leaf_blocks.extend(current)

    def _check_partition(self):
        if not self.leaf_blocks:
            raise RuntimeError("The directory has no leaf blocks.")
        covered = np.concatenate(
            [block.point_indices for block in self.leaf_blocks]
        ) if self.leaf_blocks else np.empty(0, dtype=int)
        counts = np.bincount(covered, minlength=self.data.shape[0])
        missing = np.flatnonzero(counts == 0)
        duplicated = np.flatnonzero(counts > 1)
        if counts.size != self.data.shape[0] or missing.size or duplicated.size:
            raise RuntimeError(
                "Leaf blocks do not form a partition: "
                f"{missing.size} missing, {duplicated.size} duplicated samples."
            )
        return {
            "complete": True,
            "disjoint": True,
            "n_samples": int(self.data.shape[0]),
            "n_leaf_blocks": int(len(self.leaf_blocks)),
        }

    def get_data(self):
        return self.data.tolist()

    def get_leafs(self):
        return list(self.leaf_blocks)

    def get_level(self, level):
        if isinstance(level, bool) or not isinstance(level, (int, np.integer)):
            raise TypeError("level must be an integer.")
        level = int(level)
        if level < 0:
            raise IndexError("level must be non-negative.")
        if self.observe:
            return list(self.level_blocks[level])
        if level >= self._height:
            raise IndexError("level is outside the observed directory height.")
        blocks = [self.root]
        for _ in range(level):
            blocks = [child for block in blocks for child in block.children]
            if not blocks:
                break
        return blocks

    def get_height(self):
        return self._height

    def get_root(self):
        return self.root

    def iter_all_blocks(self):
        queue = [self.root]
        while queue:
            block = queue.pop()
            yield block
            queue.extend(block.children)

    def __len__(self):
        return self._size

    def __repr__(self):
        return (
            f"BANGDirectory(levels={self.levels}, height={self._height}, "
            f"blocks={self._size}, leaves={len(self.leaf_blocks)}, "
            f"observe={self.observe})"
        )


class BANGClustering:
    """Fit-style BANG clustering API (extension of the pyclustering style)."""

    def __init__(
        self,
        levels=8,
        density_threshold=0.0,
        amount_threshold=0,
        observe=True,
    ):
        self.levels = _validate_levels(levels)
        self.density_threshold = _validate_threshold(
            density_threshold, "density_threshold"
        )
        if isinstance(amount_threshold, bool) or not isinstance(
            amount_threshold, (int, np.integer)
        ):
            raise TypeError("amount_threshold must be an integer.")
        self.amount_threshold = int(amount_threshold)
        if self.amount_threshold < 0:
            raise ValueError("amount_threshold must be non-negative.")
        self.observe = bool(observe)
        self.directory_ = None
        self.labels_ = None
        self.clusters_ = []
        self.noise_ = np.empty(0, dtype=int)
        self.cluster_blocks_ = []
        self.dendrogram_ = []
        self.processing_trace_ = []
        self.partition_check_ = {}
        self.audit_report_ = {}

    def fit(self, X, y=None):
        data = _validate_data(X)
        self.directory_ = BANGDirectory(
            data,
            self.levels,
            density_threshold=self.density_threshold,
            amount_threshold=self.amount_threshold,
            observe=self.observe,
        )
        self.partition_check_ = dict(self.directory_.partition_check)
        self._allocate_clusters()
        self._store_results()
        self.audit_report_ = self.get_audit_report()
        if not self.audit_report_["valid"]:
            raise RuntimeError(
                "BANG internal invariant failed: "
                + "; ".join(self.audit_report_["errors"])
            )
        return self

    @staticmethod
    def _block_order_key(block):
        return (-block.density, -block.creation_index)

    def _candidate_blocks(self):
        # amount_threshold controls directory growth only.  It is not used as a
        # cluster-candidate filter here; pyclustering's allocation loop does not
        # filter neighbors by size either.  density_threshold determines noise.
        return [
            block
            for block in self.directory_.leaf_blocks
            if block.density > self.density_threshold
        ]

    def _build_adjacency(self, blocks):
        count = len(blocks)
        adjacency = [[] for _ in range(count)]
        for first in range(count):
            for second in range(first + 1, count):
                if blocks[first].is_neighbor(blocks[second]):
                    adjacency[first].append(second)
                    adjacency[second].append(first)
        return adjacency

    def _allocate_clusters(self):
        """Density-ordered neighbor expansion of Sections 2.2 and 2.4.

        A cluster is seeded by the highest-density unprocessed block.  All
        unprocessed neighbors are discovered through BFS and queued in
        decreasing density order; every block appears exactly once in the
        dendrogram lists.  This follows the operational behavior of
        pyclustering.  The two local chain examples in Section 2.4 are not
        encoded as a separate global start-point algorithm because the paper
        does not provide a complete pseudocode for that choice.
        """
        for block in self.directory_.leaf_blocks:
            block.set_cluster(None)
        candidates = self._candidate_blocks()
        candidates.sort(key=self._block_order_key)
        adjacency = self._build_adjacency(candidates)
        unhandled = set(range(len(candidates)))
        cluster_blocks = []
        trace = []
        order = 0
        for seed in range(len(candidates)):
            if seed not in unhandled:
                continue
            cluster_index = len(cluster_blocks)
            queue = deque([seed])
            unhandled.remove(seed)
            discovered = []
            is_seed = True
            while queue:
                current = queue.popleft()
                block = candidates[current]
                block.set_cluster(cluster_index)
                discovered.append(block)
                trace.append(
                    {
                        "order": order,
                        "cluster": cluster_index,
                        "block": block,
                        "region": block.region,
                        "level": block.level,
                        "density": block.density,
                        "population": block.size,
                        "role": "center" if is_seed else "neighbor",
                    }
                )
                is_seed = False
                order += 1
                next_blocks = [
                    neighbor
                    for neighbor in adjacency[current]
                    if neighbor in unhandled
                ]
                next_blocks.sort(
                    key=lambda index: self._block_order_key(candidates[index])
                )
                for neighbor in next_blocks:
                    unhandled.remove(neighbor)
                    queue.append(neighbor)
            discovered.sort(key=self._block_order_key)
            cluster_blocks.append(discovered)
        self.cluster_blocks_ = cluster_blocks
        self.dendrogram_ = [list(blocks) for blocks in cluster_blocks]
        self.processing_trace_ = trace

    def _store_results(self):
        labels = np.full(self.directory_.data.shape[0], -1, dtype=int)
        assignment_count = np.zeros(self.directory_.data.shape[0], dtype=int)
        for cluster_index, blocks in enumerate(self.cluster_blocks_):
            for block in blocks:
                indices = block.point_indices
                labels[indices] = cluster_index
                assignment_count[indices] += 1
        if np.any(assignment_count > 1):
            raise RuntimeError("A sample was assigned to more than one cluster.")
        self.labels_ = labels
        self.clusters_ = [
            np.flatnonzero(labels == cluster_index)
            for cluster_index in range(len(self.cluster_blocks_))
        ]
        self.noise_ = np.flatnonzero(labels == -1)

    def get_clusters(self):
        return [cluster.tolist() for cluster in self.clusters_]

    def get_noise(self):
        return self.noise_.tolist()

    def get_labels(self):
        if self.labels_ is None:
            raise RuntimeError("The model has not been fitted yet.")
        return self.labels_.copy()

    def get_directory(self):
        return self.directory_

    def get_dendrogram(self):
        return [list(blocks) for blocks in self.dendrogram_]

    def get_cluster_blocks(self):
        return [list(blocks) for blocks in self.cluster_blocks_]

    def get_processing_trace(self):
        return [dict(item) for item in self.processing_trace_]

    def get_cluster_encoding(self):
        return ClusterEncoding.CLUSTER_INDEX_LIST_SEPARATION

    def fit_predict(self, X, y=None):
        return self.fit(X, y).labels_

    def _region_errors(self):
        errors = []
        seen = set()
        for block in self.directory_.iter_all_blocks():
            pair = (block.region, block.level)
            if pair in seen:
                errors.append(f"Duplicate region/level key {pair}.")
            seen.add(pair)
            if len(block.children) not in (0, 2):
                errors.append(f"Block {pair} has {len(block.children)} children.")
                continue
            if block.children:
                left, right = block.children
                expected_right = block.region + 2 ** block.level
                if left.region != block.region or left.level != block.level + 1:
                    errors.append(f"Left child of {pair} has incorrect region/level.")
                if right.region != expected_right or right.level != block.level + 1:
                    errors.append(f"Right child of {pair} has incorrect region/level.")
        return errors

    def get_audit_report(self):
        errors = []
        if self.directory_ is None or self.labels_ is None:
            return {
                "valid": False,
                "errors": ["The model has not been fitted."],
                "partition": {},
                "n_clusters": 0,
                "n_noise": 0,
            }
        partition = dict(self.directory_.partition_check)
        if not partition.get("complete") or not partition.get("disjoint"):
            errors.append("Leaf partition is not complete and disjoint.")
        errors.extend(self._region_errors())
        for block in self.directory_.leaf_blocks:
            expected = 0.0 if block.size == 0 else block.size / block.spatial_block.get_volume()
            if not math.isclose(block.density, expected, rel_tol=1e-12, abs_tol=1e-15):
                errors.append(
                    f"Block ({block.region},{block.level}) density is inconsistent."
                )
        candidates = self._candidate_blocks()
        candidate_ids = {id(block) for block in candidates}
        assigned = [block for cluster in self.cluster_blocks_ for block in cluster]
        if len({id(block) for block in assigned}) != len(assigned):
            errors.append("A block occurs in more than one cluster.")
        if {id(block) for block in assigned} != candidate_ids:
            errors.append("Cluster blocks do not match the candidate block set.")
        for blocks in self.cluster_blocks_:
            if blocks != sorted(blocks, key=self._block_order_key):
                errors.append("A cluster block list is not sorted by decreasing density.")
        if [list(blocks) for blocks in self.cluster_blocks_] != self.dendrogram_:
            errors.append("dendrogram_ is not the list of cluster block lists.")
        for cluster_index, blocks in enumerate(self.cluster_blocks_):
            for block in blocks:
                if block.cluster != cluster_index:
                    errors.append(
                        f"Block ({block.region},{block.level}) has wrong cluster id."
                    )
        if int(np.sum(self.labels_ < 0)) != int(self.noise_.size):
            errors.append("Noise labels and noise index array disagree.")
        for cluster_index, cluster in enumerate(self.clusters_):
            if not np.array_equal(np.flatnonzero(self.labels_ == cluster_index), cluster):
                errors.append(f"Cluster {cluster_index} index array disagrees with labels.")
        valid_labels = set(range(len(self.clusters_))) | {-1}
        if not set(int(value) for value in np.unique(self.labels_)).issubset(valid_labels):
            errors.append("labels_ contains an unknown cluster id.")
        assigned_points = np.zeros(self.directory_.data.shape[0], dtype=int)
        for blocks in self.cluster_blocks_:
            for block in blocks:
                assigned_points[block.point_indices] += 1
        if np.any(assigned_points > 1):
            errors.append("A sample occurs in more than one cluster block.")
        if len(self.processing_trace_) != len(candidates):
            errors.append("Processing trace does not cover every candidate block.")
        if self.processing_trace_:
            if [item["order"] for item in self.processing_trace_] != list(
                range(len(self.processing_trace_))
            ):
                errors.append("Processing trace order is not contiguous.")
            for item in self.processing_trace_:
                if item["role"] not in {"center", "neighbor"}:
                    errors.append("Processing trace contains an invalid role.")
        if not np.array_equal(self.noise_, np.flatnonzero(self.labels_ < 0)):
            errors.append("noise_ does not match labels_.")
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "partition": partition,
            "n_clusters": len(self.cluster_blocks_),
            "n_noise": int(self.noise_.size),
            "n_candidates": len(candidates),
            "n_leaves": len(self.directory_.leaf_blocks),
            "height": self.directory_.get_height(),
            "observed_levels": bool(self.observe),
        }


class bang:
    """pyclustering-compatible BANG facade.

    The constructor keeps the data and levels until ``process`` is called.  The
    fit-style API is available through ``BANGClustering``.
    """

    def __init__(self, data, levels, ccore=False, **kwargs):
        self._data = _validate_data(data)
        self._levels = _validate_levels(levels)
        self._ccore = bool(ccore)
        self._kwargs = dict(kwargs)
        self._model = None

    def process(self):
        self._model = BANGClustering(levels=self._levels, **self._kwargs)
        self._model.fit(self._data)
        return self

    def _require_model(self):
        if self._model is None:
            raise RuntimeError("Call process() before reading clustering results.")
        return self._model

    def get_clusters(self):
        if self._model is None:
            return []
        return self._model.get_clusters()

    def get_noise(self):
        if self._model is None:
            return []
        return self._model.get_noise()

    def get_directory(self):
        if self._model is None:
            return None
        return self._model.get_directory()

    def get_dendrogram(self):
        if self._model is None:
            return []
        return self._model.get_dendrogram()

    def get_cluster_encoding(self):
        return ClusterEncoding.CLUSTER_INDEX_LIST_SEPARATION

    def get_cluster_blocks(self):
        if self._model is None:
            return []
        return self._model.get_cluster_blocks()

    def get_processing_trace(self):
        if self._model is None:
            return []
        return self._model.get_processing_trace()

    def get_audit_report(self):
        if self._model is None:
            return {"valid": False, "errors": ["The model has not been processed."]}
        return self._model.get_audit_report()

    def __repr__(self):
        state = "unprocessed" if self._model is None else "processed"
        return f"bang(levels={self._levels}, state={state})"


spatial_block = SpatialBlock
bang_block = BANGBlock
bang_directory = BANGDirectory


def min_max_normalize(data):
    """Scale every feature to [0, 1]; constant features become zero.

    This is a convenience preprocessing function, not part of BANG-Clustering.
    """
    data = _validate_data(data)
    low = np.min(data, axis=0)
    span = np.max(data, axis=0) - low
    span = np.where(span > 0.0, span, 1.0)
    return (data - low) / span


class BANGVisualizer:
    """Matplotlib visualization helpers.

    These methods are implementation extensions.  The static compatibility
    facade ``bang_visualizer`` exposes the methods documented by pyclustering.
    """

    def __init__(
        self,
        figsize=(16.0, 5.0),
        point_size=18.0,
        point_alpha=0.82,
        block_alpha=0.48,
        cmap="YlGnBu",
    ):
        self.figsize = figsize
        self.point_size = float(point_size)
        self.point_alpha = float(point_alpha)
        self.block_alpha = float(block_alpha)
        self.cmap = cmap

    def _resolve_axes(self, n_features, feature_indices):
        if n_features == 1:
            return 0, None
        if feature_indices is None:
            feature_indices = (0, 1)
        if len(feature_indices) != 2:
            raise ValueError("feature_indices must contain two dimensions.")
        first, second = (int(feature_indices[0]), int(feature_indices[1]))
        if first < 0 or second < 0 or first >= n_features or second >= n_features:
            raise ValueError("feature_indices is out of range.")
        if first == second:
            raise ValueError("feature_indices must contain two different dimensions.")
        return first, second

    def _coordinates(self, data, feature_indices):
        data = np.asarray(data, dtype=float)
        first, second = self._resolve_axes(data.shape[1], feature_indices)
        x = data[:, first]
        y = np.zeros(data.shape[0]) if second is None else data[:, second]
        return x, y, first, second

    def _rectangle(self, block, first, second):
        max_corner = block.spatial_block.max_corner
        min_corner = block.spatial_block.min_corner
        x0 = float(min_corner[first])
        width = float(max_corner[first] - min_corner[first])
        if second is None:
            y0, height = -0.5, 1.0
        else:
            y0 = float(min_corner[second])
            height = float(max_corner[second] - min_corner[second])
        return x0, y0, width, height

    @staticmethod
    def _cluster_color(index):
        return plt.get_cmap("tab10")(int(index) % 10)

    def _set_limits(self, axis, directory, first, second):
        data = directory.data
        x = data[:, first]
        x_pad = max(float(np.ptp(x)) * 0.05, 1e-9)
        axis.set_xlim(float(np.min(x)) - x_pad, float(np.max(x)) + x_pad)
        if second is None:
            axis.set_ylim(-0.7, 0.7)
        else:
            y = data[:, second]
            y_pad = max(float(np.ptp(y)) * 0.05, 1e-9)
            axis.set_ylim(float(np.min(y)) - y_pad, float(np.max(y)) + y_pad)

    def plot_blocks(self, directory, ax=None, feature_indices=(0, 1), colorbar=True):
        if not isinstance(directory, BANGDirectory):
            raise TypeError("directory must be a BANGDirectory.")
        if ax is None:
            _, ax = plt.subplots(figsize=(7.0, 6.0))
        first, second = self._resolve_axes(directory.data.shape[1], feature_indices)
        x, y, _, _ = self._coordinates(directory.data, feature_indices)
        densities = np.array([block.density for block in directory.get_leafs()])
        vmin = float(np.min(densities)) if densities.size else 0.0
        vmax = float(np.max(densities)) if densities.size else 1.0
        if vmax <= vmin:
            vmax = vmin + 1.0
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap(self.cmap)
        for block in directory.get_leafs():
            x0, y0, width, height = self._rectangle(block, first, second)
            ax.add_patch(
                patches.Rectangle(
                    (x0, y0),
                    width,
                    height,
                    facecolor=cmap(norm(block.density)),
                    edgecolor="#263238",
                    linewidth=0.45,
                    alpha=self.block_alpha,
                )
            )
        ax.scatter(x, y, s=4.0, color="#111111", alpha=0.55, linewidths=0.0, zorder=3)
        self._set_limits(ax, directory, first, second)
        ax.set_xlabel(f"Feature {first}")
        ax.set_ylabel("Value" if second is None else f"Feature {second}")
        ax.set_title("BANG blocks")
        ax.grid(False)
        if colorbar:
            scalar_map = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            scalar_map.set_array(densities)
            plt.colorbar(scalar_map, ax=ax, fraction=0.046, pad=0.04, label="Density")
        return ax

    def plot_clusters(
        self,
        data,
        labels,
        directory=None,
        ax=None,
        feature_indices=(0, 1),
        show_noise=True,
    ):
        data = np.asarray(data, dtype=float)
        labels = np.asarray(labels, dtype=int)
        if labels.shape[0] != data.shape[0]:
            raise ValueError("labels and data must have the same number of samples.")
        if directory is not None and directory.data.shape != data.shape:
            raise ValueError("data and directory.data must have the same shape.")
        if ax is None:
            _, ax = plt.subplots(figsize=(7.0, 6.0))
        x, y, first, second = self._coordinates(data, feature_indices)
        for cluster_index in sorted(int(v) for v in np.unique(labels) if v >= 0):
            mask = labels == cluster_index
            ax.scatter(
                x[mask],
                y[mask],
                s=self.point_size,
                color=self._cluster_color(cluster_index),
                alpha=self.point_alpha,
                edgecolors="none",
                label=f"Cluster {cluster_index}",
                zorder=4,
            )
        if show_noise and np.any(labels < 0):
            mask = labels < 0
            ax.scatter(
                x[mask],
                y[mask],
                s=self.point_size * 1.3,
                color="#7a7a7a",
                marker="x",
                linewidths=0.9,
                alpha=0.85,
                label="Noise",
                zorder=5,
            )
        if directory is not None:
            for block in directory.get_leafs():
                x0, y0, width, height = self._rectangle(block, first, second)
                edgecolor = (
                    "#b0bec5"
                    if block.cluster is None
                    else self._cluster_color(block.cluster)
                )
                ax.add_patch(
                    patches.Rectangle(
                        (x0, y0),
                        width,
                        height,
                        fill=False,
                        edgecolor=edgecolor,
                        linewidth=0.7 if block.cluster is not None else 0.35,
                        alpha=0.85,
                        zorder=2,
                    )
                )
            self._set_limits(ax, directory, first, second)
        else:
            x_pad = max(float(np.ptp(x)) * 0.05, 1e-9)
            y_pad = max(float(np.ptp(y)) * 0.05, 1e-9)
            ax.set_xlim(float(np.min(x)) - x_pad, float(np.max(x)) + x_pad)
            ax.set_ylim(float(np.min(y)) - y_pad, float(np.max(y)) + y_pad)
        ax.set_xlabel(f"Feature {first}")
        ax.set_ylabel("Value" if second is None else f"Feature {second}")
        ax.set_title("BANG clusters")
        ax.grid(False)
        if np.any(labels >= 0) or (show_noise and np.any(labels < 0)):
            ax.legend(loc="best", frameon=True, fontsize=8)
        return ax

    def plot_dendrogram(self, dendrogram, ax=None):
        if dendrogram is None:
            raise ValueError("dendrogram must not be None.")
        if ax is None:
            _, ax = plt.subplots(figsize=(8.0, 4.5))
        position = 0
        for cluster_index, blocks in enumerate(dendrogram):
            if not blocks:
                continue
            densities = np.array([block.density for block in blocks])
            positions = np.arange(position, position + len(blocks))
            ax.bar(
                positions,
                densities,
                width=0.9,
                color=self._cluster_color(cluster_index),
                alpha=0.85,
                linewidth=0.0,
                label=f"Cluster {cluster_index}",
            )
            position += len(blocks)
        ax.set_xlabel("BANG block")
        ax.set_ylabel("Density")
        ax.set_title("BANG dendrogram")
        ax.set_xticks([])
        ax.grid(axis="y", alpha=0.25)
        if position > 0:
            ax.set_xlim(-0.7, position - 0.3)
        if dendrogram:
            ax.legend(loc="best", fontsize=8)
        return ax

    def plot_all(self, data, labels, directory, dendrogram, feature_indices=(0, 1)):
        figure, axes = plt.subplots(1, 3, figsize=self.figsize)
        self.plot_blocks(directory, ax=axes[0], feature_indices=feature_indices, colorbar=False)
        self.plot_clusters(
            data,
            labels,
            directory=directory,
            ax=axes[1],
            feature_indices=feature_indices,
        )
        self.plot_dendrogram(dendrogram, ax=axes[2])
        figure.tight_layout()
        return figure


class bang_visualizer:
    """Static compatibility facade for pyclustering-style visualization."""

    @staticmethod
    def _dimension_pairs(dimension):
        if dimension <= 1:
            return [(0, None)]
        return [
            (first, second)
            for first in range(dimension)
            for second in range(first + 1, dimension)
        ]

    @staticmethod
    def show_blocks(directory):
        if not isinstance(directory, BANGDirectory):
            raise TypeError("directory must be a BANGDirectory.")
        import itertools

        dimension = directory.data.shape[1]
        pairs = list(itertools.combinations(range(dimension), 2))
        if not pairs:
            pairs = [(0, 0)]
        visualizer = BANGVisualizer(figsize=(5.5 * len(pairs), 5.0))
        figure, axes = plt.subplots(1, len(pairs), figsize=(5.5 * len(pairs), 5.0))
        if len(pairs) == 1:
            axes = [axes]
        for index, (first, second) in enumerate(pairs):
            feature_indices = None if first == second else (first, second)
            visualizer.plot_blocks(
                directory,
                ax=axes[index],
                feature_indices=feature_indices,
                colorbar=index == 0,
            )
        plt.tight_layout()
        plt.show()
        return None

    @staticmethod
    def show_dendrogram(dendrogram):
        visualizer = BANGVisualizer()
        figure, axis = plt.subplots(figsize=(8.0, 4.5))
        visualizer.plot_dendrogram(dendrogram, ax=axis)
        plt.show()
        return None

    @staticmethod
    def show_clusters(data, clusters, noise=None):
        import itertools

        data = np.asarray(data, dtype=float)
        labels = np.full(data.shape[0], -1, dtype=int)
        for cluster_index, cluster in enumerate(clusters):
            labels[np.asarray(cluster, dtype=int)] = cluster_index
        if noise is not None:
            labels[np.asarray(noise, dtype=int)] = -1
        dimension = data.shape[1]
        pairs = list(itertools.combinations(range(dimension), 2))
        if not pairs:
            pairs = [(0, 0)]
        visualizer = BANGVisualizer(figsize=(5.5 * len(pairs), 5.0))
        figure, axes = plt.subplots(1, len(pairs), figsize=(5.5 * len(pairs), 5.0))
        if len(pairs) == 1:
            axes = [axes]
        for index, (first, second) in enumerate(pairs):
            feature_indices = None if first == second else (first, second)
            visualizer.plot_clusters(
                data,
                labels,
                ax=axes[index],
                feature_indices=feature_indices,
            )
        plt.tight_layout()
        plt.show()
        return None


class bang_animator:
    """Animate the directory and the density-ordered clustering process.

    The public constructor and ``animate`` signature follow pyclustering.  Two
    optional keyword arguments extend it: ``noise`` (for the final frame) and
    ``feature_indices`` (to project data with dimension > 2).
    """

    def __init__(
        self,
        directory,
        clusters,
        noise=None,
        feature_indices=(0, 1),
        processing_trace=None,
    ):
        if not isinstance(directory, BANGDirectory):
            raise TypeError("directory must be a BANGDirectory.")
        data = directory.data
        if data.shape[1] < 2:
            raise ValueError("Animation requires at least two features.")
        self._directory = directory
        self._feature_indices = tuple(feature_indices)
        self._clusters = [np.asarray(cluster, dtype=int) for cluster in clusters]
        for cluster in self._clusters:
            if np.any(cluster < 0) or np.any(cluster >= data.shape[0]):
                raise ValueError("cluster indices are out of range.")
        clustered = set()
        for cluster in self._clusters:
            clustered.update(int(index) for index in cluster)
        if noise is None:
            noise = sorted(set(range(data.shape[0])) - clustered)
        noise = np.asarray(sorted(set(int(index) for index in noise)), dtype=int)
        if noise.size and (np.any(noise < 0) or np.any(noise >= data.shape[0])):
            raise ValueError("noise indices are out of range.")
        self._noise = np.asarray(sorted(set(noise.tolist()) - clustered), dtype=int)
        self._label = np.full(data.shape[0], -1, dtype=int)
        for cluster_index, cluster in enumerate(self._clusters):
            self._label[cluster] = cluster_index
        self._leafs = sorted(
            directory.get_leafs(),
            key=lambda block: (-block.density, -block.creation_index),
        )
        if processing_trace:
            ordered = [item["block"] for item in processing_trace]
            ordered_ids = {id(block) for block in ordered}
            if len(ordered_ids) != len(ordered):
                raise ValueError("processing_trace contains duplicate blocks.")
            missing = [block for block in self._leafs if id(block) not in ordered_ids]
            self._ordered_blocks = ordered + missing
        else:
            self._ordered_blocks = list(self._leafs)
        self._leaf_cluster = {}
        for block in self._leafs:
            indices = block.point_indices
            if indices.size == 0:
                self._leaf_cluster[id(block)] = None
                continue
            values = np.unique(self._label[indices])
            self._leaf_cluster[id(block)] = int(values[0]) if len(values) == 1 and values[0] >= 0 else None
        self._coordinates()
        self._sequence = self._build_sequence()
        self._figure, self._axis = plt.subplots(figsize=(8.0, 6.0))
        self._animation = None

    def __del__(self):
        try:
            plt.close(self._figure)
        except Exception:
            pass

    def _build_sequence(self):
        sequence = [("data",)]
        for level in range(self._directory.get_height()):
            if self._directory.get_level(level):
                sequence.append(("level", level))
        sequence.append(("density",))
        for block in self._ordered_blocks:
            sequence.append(("block", block))
        sequence.append(("final",))
        return sequence

    def _coordinates(self):
        data = self._directory.data
        first, second = self._feature_indices
        if first == second or first < 0 or second < 0 or first >= data.shape[1] or second >= data.shape[1]:
            raise ValueError("feature_indices is out of range.")
        return data[:, first], data[:, second], first, second

    def _draw_block(self, block, facecolor="none", edgecolor="black", alpha=1.0, linewidth=0.7):
        x, y, first, second = self._coordinates()
        max_corner = block.spatial_block.max_corner
        min_corner = block.spatial_block.min_corner
        x0 = float(min_corner[first])
        y0 = float(min_corner[second])
        width = float(max_corner[first] - min_corner[first])
        height = float(max_corner[second] - min_corner[second])
        self._axis.add_patch(
            patches.Rectangle(
                (x0, y0),
                width,
                height,
                facecolor=facecolor,
                edgecolor=edgecolor,
                alpha=alpha,
                linewidth=linewidth,
                zorder=2,
            )
        )

    def _setup_axis(self):
        self._axis.clear()
        x, y, first, second = self._coordinates()
        self._axis.set_xlabel(f"Feature {first}")
        self._axis.set_ylabel(f"Feature {second}")
        self._axis.grid(False)
        self._axis.scatter(x, y, s=3.0, color="#888888", alpha=0.35, linewidths=0.0, zorder=1)
        self._axis.set_xlim(float(np.min(x)) - 0.1, float(np.max(x)) + 0.1)
        self._axis.set_ylim(float(np.min(y)) - 0.1, float(np.max(y)) + 0.1)

    def _frame(self, frame_index):
        stage, *payload = self._sequence[frame_index]
        self._setup_axis()
        if stage == "data":
            self._axis.set_title("1. Input data")
        elif stage == "level":
            level = payload[0]
            self._axis.set_title(f"2. Directory level {level}")
            for block in self._directory.get_level(level):
                self._draw_block(block, facecolor="none", edgecolor="#263238", alpha=0.9)
        elif stage == "density":
            self._axis.set_title("3. Leaf density")
            densities = np.array([block.density for block in self._leafs])
            vmax = float(np.max(densities)) if densities.size else 1.0
            if vmax <= 0.0:
                vmax = 1.0
            for block in self._leafs:
                alpha = 0.75 * block.density / vmax
                self._draw_block(
                    block,
                    facecolor=mcolors.to_rgba("tab:blue", alpha=alpha),
                    edgecolor="#263238",
                    alpha=1.0,
                )
        elif stage == "block":
            block = payload[0]
            current_index = self._ordered_blocks.index(block)
            processed = set(self._ordered_blocks[: current_index + 1])
            self._axis.set_title(
                f"4. Processing block ({block.region}, {block.level})"
            )
            for leaf in self._leafs:
                if leaf in processed:
                    cluster = self._leaf_cluster[id(leaf)]
                    color = (
                        "#9e9e9e"
                        if cluster is None
                        else BANGVisualizer._cluster_color(cluster)
                    )
                    self._draw_block(
                        leaf,
                        facecolor=mcolors.to_rgba(color, alpha=0.55),
                        edgecolor="black",
                        linewidth=0.5,
                    )
                else:
                    self._draw_block(
                        leaf,
                        facecolor="none",
                        edgecolor="#cfd8dc",
                        alpha=0.7,
                        linewidth=0.35,
                    )
            self._draw_block(
                block,
                facecolor="none",
                edgecolor="red",
                linewidth=1.8,
            )
            indices = block.point_indices
            if indices.size:
                x, y, first, second = self._coordinates()
                colors = [
                    "#9e9e9e"
                    if self._label[index] < 0
                    else BANGVisualizer._cluster_color(self._label[index])
                    for index in indices
                ]
                self._axis.scatter(
                    x[indices], y[indices], s=10.0, c=colors, zorder=4
                )
        elif stage == "final":
            self._axis.set_title("5. Final clusters and noise")
            for block in self._leafs:
                self._draw_block(
                    block,
                    facecolor="none",
                    edgecolor="#b0bec5",
                    alpha=0.5,
                    linewidth=0.35,
                )
            x, y, first, second = self._coordinates()
            for cluster_index, cluster in enumerate(self._clusters):
                self._axis.scatter(
                    x[cluster],
                    y[cluster],
                    s=12.0,
                    color=BANGVisualizer._cluster_color(cluster_index),
                    alpha=0.9,
                    zorder=4,
                    label=f"Cluster {cluster_index}",
                )
            if self._noise.size:
                self._axis.scatter(
                    x[self._noise],
                    y[self._noise],
                    s=15.0,
                    color="#7a7a7a",
                    marker="x",
                    linewidths=0.8,
                    zorder=5,
                    label="Noise",
                )
            self._axis.legend(loc="best", fontsize=8)
        return []

    def animate(
        self,
        animation_velocity=75,
        movie_fps=25,
        movie_filename=None,
    ):
        if animation_velocity <= 0:
            raise ValueError("animation_velocity must be positive.")
        if movie_fps <= 0:
            raise ValueError("movie_fps must be positive.")
        self._animation = animation.FuncAnimation(
            self._figure,
            self._frame,
            frames=len(self._sequence),
            interval=animation_velocity,
            repeat=False,
        )
        if movie_filename is not None:
            suffix = str(movie_filename).lower()
            if suffix.endswith(".gif"):
                writer = "pillow"
            elif suffix.endswith((".mp4", ".m4v", ".mov")):
                writer = "ffmpeg"
            else:
                raise ValueError("movie_filename must end with .gif or .mp4.")
            if not animation.writers.is_available(writer):
                raise RuntimeError(
                    f"Matplotlib writer '{writer}' is not available in this environment."
                )
            self._animation.save(movie_filename, writer=writer, fps=movie_fps)
        else:
            plt.show()
        return self._animation


def run_self_tests(verbose=True):
    """Run deterministic checks for the core invariants.

    This is a test/demo helper, not part of the clustering algorithm.  It
    returns the number of checks that passed.
    """
    checks = []

    def check(condition, message):
        if not condition:
            raise AssertionError(message)
        checks.append(message)

    rng = np.random.default_rng(2026)
    data = rng.normal(size=(40, 2))

    observed = BANGDirectory(data, 4, observe=True)
    lazy = BANGDirectory(data, 4, observe=False)
    check(observed.get_height() == lazy.get_height(), "observe modes have equal height")
    for level in range(observed.get_height()):
        left = [(block.region, block.level) for block in observed.get_level(level)]
        right = [(block.region, block.level) for block in lazy.get_level(level)]
        check(left == right, f"lazy get_level({level}) matches observed levels")

    root = observed.get_root()
    check((root.region, root.level) == (0, 0), "root region is (0, 0)")
    for block in observed.iter_all_blocks():
        if block.children:
            left, right = block.children
            check(left.region == block.region, "left child keeps parent region")
            check(
                right.region == block.region + 2 ** block.level,
                "right child follows BANG region numbering",
            )
            check(
                left.level == right.level == block.level + 1,
                "children increase level by one",
            )

    for block in observed.get_leafs():
        expected = 0.0 if block.size == 0 else block.size / block.spatial_block.get_volume()
        check(
            math.isclose(block.density, expected, rel_tol=1e-12, abs_tol=1e-15),
            "leaf density equals population / volume",
        )

    datasets = [
        rng.normal(size=(30, 1)),
        rng.normal(size=(30, 3)),
        np.c_[np.arange(12.0), np.zeros(12)],
        np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
        np.array([[0.5, 0.5]] * 8 + [[1.5, 1.5]] * 8),
        np.ones((7, 2)),
        np.array([[1.0, 2.0]]),
    ]
    for data_case in datasets:
        for levels in (1, 2, 5):
            model = BANGClustering(levels=levels, observe=True).fit(data_case)
            check(np.all(model.labels_ >= 0), "zero-threshold fit clusters every point")
            check(
                model.get_directory().partition_check["complete"]
                and model.get_directory().partition_check["disjoint"],
                "leaf partition is complete and disjoint",
            )
            for blocks in model.get_cluster_blocks():
                densities = [block.density for block in blocks]
                check(densities == sorted(densities, reverse=True), "cluster blocks are density-sorted")
            check(model.get_audit_report()["valid"], "audit passes")

    face_a = SpatialBlock([1.0, 1.0], [0.0, 0.0])
    face_b = SpatialBlock([2.0, 1.0], [1.0, 0.0])
    corner_b = SpatialBlock([2.0, 2.0], [1.0, 1.0])
    gap_b = SpatialBlock([2.5, 1.0], [2.0, 0.0])
    check(face_a.is_neighbor(face_b), "face-adjacent blocks are neighbors")
    check(face_a.neighbor_degree(face_b) == 1, "2-D face contact has degree 1")
    check(face_a.is_neighbor(corner_b), "corner-adjacent blocks are neighbors")
    check(face_a.neighbor_degree(corner_b) == 0, "2-D corner contact has degree 0")
    check(not face_a.is_neighbor(gap_b), "separated blocks are not neighbors")
    check(not face_a.is_neighbor(SpatialBlock([2.5, 2.5], [0.5, 0.5])), "nested boxes are not boundary neighbors")

    face_3d = SpatialBlock([2.0, 1.0, 1.0], [1.0, 0.0, 0.0])
    edge_3d = SpatialBlock([2.0, 2.0, 1.0], [1.0, 1.0, 0.0])
    corner_3d = SpatialBlock([2.0, 2.0, 2.0], [1.0, 1.0, 1.0])
    base_3d = SpatialBlock([1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
    check(base_3d.neighbor_degree(face_3d) == 2, "3-D face contact has degree 2")
    check(base_3d.neighbor_degree(edge_3d) == 1, "3-D edge contact has degree 1")
    check(base_3d.neighbor_degree(corner_3d) == 0, "3-D corner contact has degree 0")

    big = SpatialBlock([2.0, 1.0], [0.0, 0.0])
    small = SpatialBlock([3.0, 0.5], [2.0, 0.0])
    check(big.neighbor_degree(small) == 1, "different-level face contact is detected")

    left_blob = rng.normal([0.0, 0.0], 0.5, (40, 2))
    right_blob = rng.normal([5.0, 5.0], 0.5, (40, 2))
    blobs = np.vstack([left_blob, right_blob])
    model = BANGClustering(levels=6, observe=True).fit(blobs)
    check(len(model.get_clusters()) == 2, "two separated blobs form two clusters")
    check(len(model.get_noise()) == 0, "two clean blobs have no noise")
    check(
        len(model.get_dendrogram()) == len(model.get_clusters()) == 2,
        "dendrogram has one block list per cluster",
    )
    check(
        [len(cluster) for cluster in model.get_clusters()] == [40, 40],
        "each blob has 40 clustered samples",
    )

    symmetric = np.array([[-2.0, -2.0], [-2.0, 2.0], [2.0, -2.0], [2.0, 2.0]])
    tied = BANGClustering(levels=3).fit(symmetric)
    check(tied.get_audit_report()["valid"], "equal-density blocks are handled deterministically")

    one_block = BANGClustering(levels=1).fit(blobs)
    check(len(one_block.get_clusters()) == 1, "levels=1 returns one cluster")
    all_noise = BANGClustering(levels=3, density_threshold=1e9).fit(blobs)
    check(len(all_noise.get_clusters()) == 0, "high density threshold can produce only noise")
    check(len(all_noise.get_noise()) == 80, "all samples are noise in the high-threshold case")

    compat = bang(blobs.tolist(), 6)
    check(compat.get_directory() is None, "bang directory is unset before process")
    check(compat.get_clusters() == [], "bang clusters are empty before process")
    compat.process()
    check(compat.get_cluster_encoding() == ClusterEncoding.CLUSTER_INDEX_LIST_SEPARATION,
          "bang exposes the pyclustering cluster encoding")
    check(len(compat.get_clusters()) == 2, "bang facade produces two clusters")
    check(compat.get_dendrogram() == compat.get_cluster_blocks(), "bang dendrogram is block lists")

    lazy_model = BANGClustering(levels=4, observe=False).fit(blobs)
    check(lazy_model.get_directory().get_height() >= 2, "observe=False still computes height")
    check(all(model.get_audit_report()["valid"] for model in [
        BANGClustering(levels=4, observe=False).fit(blobs)
    ]), "observe=False audit passes")

    animator = bang_animator(
        model.get_directory(),
        model.get_clusters(),
        noise=model.get_noise(),
    )
    check(len(animator._sequence) >= 4, "animator builds multiple stages")
    traced = BANGClustering(levels=4).fit(blobs)
    traced_animator = bang_animator(
        traced.get_directory(),
        traced.get_clusters(),
        noise=traced.get_noise(),
        processing_trace=traced.get_processing_trace(),
    )
    check(
        traced_animator._ordered_blocks[0] is traced.get_processing_trace()[0]["block"],
        "animator can follow the clustering processing trace",
    )
    plt.close(traced_animator._figure)
    animator._frame(0)
    animator._frame(len(animator._sequence) - 1)
    plt.close(animator._figure)

    if verbose:
        print(f"BANG self-tests: {len(checks)} checks passed")
    return len(checks)


BANGAnimator = bang_animator


if __name__ == "__main__":
    run_self_tests(verbose=True)
