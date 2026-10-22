#!/usr/bin/env python3
"""
fusion_node.py
--------------
Camera-LiDAR frustum fusion.

For each LiDAR scan:
  1. Pull the latest Detection2DArray from each camera that is within the
     sync slop window of the LiDAR stamp.
  2. For each 2D box, transform LiDAR points into that camera's optical
     frame, project to pixels, keep points that land inside the box AND
     have positive depth (in front of camera). This is the frustum.
  3. Cluster the frustum points with DBSCAN and take the nearest cluster;
     this rejects background points that happen to fall inside the box
     (the sky through a cone, a wall behind a person, etc).
  4. Compute the cluster centroid + axis-aligned extent -> 3D box in the
     LiDAR frame. The LiDAR owns the range; the camera owns the class.
  5. Publish:
       - /fusion/objects   : custom Detection3DArray with class + 3D box
       - /fusion/points_labeled : original cloud with per-point object_id
       - /fusion/markers   : MarkerArray for RViz (handled by viz_node)

The viz node is separate so this stays headless-friendly on the Jetson
(you can run fusion over SSH, pop RViz open on the monitor later).

Why no neural fusion model: PROVE's ODD is static cones on a flat lot.
Classical frustum fusion is the textbook accurate approach for that; a
BEVFusion-style net is overkill and harder to debug when calibration
drifts. If you later get moving obstacles, swap this out.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Pose, Vector3
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Header
from vision_msgs.msg import (
    BoundingBox3D,
    Detection2DArray,
    Detection3D,
    Detection3DArray,
    ObjectHypothesisWithPose,
)


# ---------------------------------------------------------------- helpers --

def _mat_from_param(flat_or_nested) -> np.ndarray:
    """Accept either a 16-element flat list or a 4x4 nested list from params."""
    arr = np.asarray(flat_or_nested, dtype=np.float64)
    if arr.shape == (16,):
        arr = arr.reshape(4, 4)
    if arr.shape != (4, 4):
        raise ValueError(f'Expected 4x4 transform, got shape {arr.shape}')
    return arr


def _invert_transform(T: np.ndarray) -> np.ndarray:
    """Fast inverse of a rigid transform (R, t) -> (R^T, -R^T t)."""
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=T.dtype)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def _dbscan_largest_nearest_cluster(
    pts: np.ndarray, eps: float, min_samples: int, min_cluster: int
) -> Optional[np.ndarray]:
    """Run DBSCAN and return the cluster whose centroid is closest to origin.

    Why nearest, not largest: a cone in the foreground + a wall behind it
    might both land inside the 2D box. We want the cone. 'Nearest to the
    camera' matches that intuition; 'largest' would pick the wall.

    Returns None if nothing passes the min_cluster size.
    """
    if len(pts) < min_samples:
        return None
    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        # Degenerate fallback: single cluster of everything.
        return pts if len(pts) >= min_cluster else None

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts)
    best = None
    best_range = np.inf
    for lbl in set(labels):
        if lbl == -1:
            continue
        mask = labels == lbl
        if mask.sum() < min_cluster:
            continue
        cluster = pts[mask]
        centroid = cluster.mean(axis=0)
        r = float(np.linalg.norm(centroid[:2]))  # XY range from origin
        if r < best_range:
            best_range = r
            best = cluster
    return best


# ---------------------------------------------------------------- camera --
@dataclass
class CameraRuntime:
    name: str
    frame_id: str
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    T_lidar_cam: np.ndarray    # 4x4, point_lidar = T * point_cam
    T_cam_lidar: np.ndarray    # inverse, point_cam = T * point_lidar
    # Rolling buffer of recent detections (stamp_ns, Detection2DArray)
    det_buffer: Deque[Tuple[int, Detection2DArray]]


# ---------------------------------------------------------------- node ----
class FusionNode(Node):

    def __init__(self):
        super().__init__('prove_fusion')

        # ---- Params ---------------------------------------------------------
        self.declare_parameter('lidar.frame_id', 'lidar_link')
        self.declare_parameter('lidar.topic', '/lidar/points')
        self.declare_parameter('lidar.min_range', 0.5)
        self.declare_parameter('lidar.max_range', 80.0)

        self.declare_parameter('fusion.sync_slop_sec', 0.05)
        self.declare_parameter('fusion.min_frustum_points', 8)
        self.declare_parameter('fusion.dbscan_eps', 0.5)
        self.declare_parameter('fusion.dbscan_min_samples', 5)
        self.declare_parameter('fusion.min_cluster_points', 10)
        self.declare_parameter('fusion.max_object_range', 40.0)
        self.declare_parameter('fusion.min_object_range', 0.5)

        # Per-camera params (same keys as detector)
        for name in ('front', 'left', 'right'):
            self.declare_parameter(f'cameras.{name}.frame_id', f'cam_{name}_optical')
            self.declare_parameter(f'cameras.{name}.fx', 640.0)
            self.declare_parameter(f'cameras.{name}.fy', 640.0)
            self.declare_parameter(f'cameras.{name}.cx', 640.0)
            self.declare_parameter(f'cameras.{name}.cy', 360.0)
            self.declare_parameter(f'cameras.{name}.width', 1280)
            self.declare_parameter(f'cameras.{name}.height', 720)
            self.declare_parameter(
                f'cameras.{name}.T_lidar_cam',
                [1.0, 0.0, 0.0, 0.0,
                 0.0, 1.0, 0.0, 0.0,
                 0.0, 0.0, 1.0, 0.0,
                 0.0, 0.0, 0.0, 1.0],
            )

        self.lidar_frame = self.get_parameter('lidar.frame_id').value
        self.min_range = float(self.get_parameter('lidar.min_range').value)
        self.max_range = float(self.get_parameter('lidar.max_range').value)
        self.slop = float(self.get_parameter('fusion.sync_slop_sec').value)
        self.min_frustum_pts = int(self.get_parameter('fusion.min_frustum_points').value)
        self.dbscan_eps = float(self.get_parameter('fusion.dbscan_eps').value)
        self.dbscan_min_samples = int(self.get_parameter('fusion.dbscan_min_samples').value)
        self.min_cluster = int(self.get_parameter('fusion.min_cluster_points').value)
        self.max_obj_r = float(self.get_parameter('fusion.max_object_range').value)
        self.min_obj_r = float(self.get_parameter('fusion.min_object_range').value)

        self.cameras: Dict[str, CameraRuntime] = {}
        for name in ('front', 'left', 'right'):
            T = _mat_from_param(self.get_parameter(f'cameras.{name}.T_lidar_cam').value)
            self.cameras[name] = CameraRuntime(
                name=name,
                frame_id=self.get_parameter(f'cameras.{name}.frame_id').value,
                fx=float(self.get_parameter(f'cameras.{name}.fx').value),
                fy=float(self.get_parameter(f'cameras.{name}.fy').value),
                cx=float(self.get_parameter(f'cameras.{name}.cx').value),
                cy=float(self.get_parameter(f'cameras.{name}.cy').value),
                width=int(self.get_parameter(f'cameras.{name}.width').value),
                height=int(self.get_parameter(f'cameras.{name}.height').value),
                T_lidar_cam=T,
                T_cam_lidar=_invert_transform(T),
                det_buffer=deque(maxlen=20),
            )

        # ---- IO -------------------------------------------------------------
        qos_sens = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        lidar_topic = self.get_parameter('lidar.topic').value
        self.create_subscription(PointCloud2, lidar_topic, self._on_cloud, qos_sens)
        self.get_logger().info(f'subscribed {lidar_topic}')

        for name, cam in self.cameras.items():
            det_topic = f'/detections/{name}'
            self.create_subscription(
                Detection2DArray, det_topic,
                lambda msg, n=name: self._on_det(n, msg),
                qos_sens,
            )
            self.get_logger().info(f'[{name}] subscribed {det_topic}')

        self.pub_obj = self.create_publisher(Detection3DArray, '/fusion/objects', 10)
        self.pub_pts = self.create_publisher(PointCloud2, '/fusion/points_labeled', 10)
        self.get_logger().info('fusion node ready')

    # ---------------------------------------------------------------- subs --
    def _on_det(self, cam_name: str, msg: Detection2DArray) -> None:
        # Stamp as integer nanoseconds for easy comparison with LiDAR stamps.
        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        self.cameras[cam_name].det_buffer.append((stamp_ns, msg))

    def _find_matching_det(
        self, cam: CameraRuntime, lidar_stamp_ns: int
    ) -> Optional[Detection2DArray]:
        """Return the camera detection closest in time to this LiDAR stamp."""
        if not cam.det_buffer:
            return None
        slop_ns = int(self.slop * 1e9)
        best = None
        best_dt = slop_ns + 1
        for stamp, msg in cam.det_buffer:
            dt = abs(stamp - lidar_stamp_ns)
            if dt < best_dt:
                best_dt = dt
                best = msg
        return best if best_dt <= slop_ns else None

    # ---------------------------------------------------------------- main --
    def _on_cloud(self, cloud_msg: PointCloud2) -> None:
        """Fuse one LiDAR scan with the latest matching camera detections."""
        lidar_stamp_ns = (
            cloud_msg.header.stamp.sec * 1_000_000_000
            + cloud_msg.header.stamp.nanosec
        )

        # Pull xyz from the cloud. field_names=('x','y','z') skips intensity
        # and anything else; we don't need those for fusion.
        pts_list = list(pc2.read_points(
            cloud_msg, field_names=('x', 'y', 'z'), skip_nans=True))
        if not pts_list:
            return
        # Each row is a tuple-like; stacking into float32 is faster than
        # per-point access later.
        pts = np.asarray(
            [(p[0], p[1], p[2]) for p in pts_list], dtype=np.float32)

        # Range filter (near-field dead zone + max range gate)
        ranges = np.linalg.norm(pts[:, :2], axis=1)
        in_range = (ranges >= self.min_range) & (ranges <= self.max_range)
        pts = pts[in_range]
        if len(pts) == 0:
            return

        # Per-point object label; -1 means "background / unlabeled".
        labels = np.full(len(pts), -1, dtype=np.int32)

        out = Detection3DArray()
        out.header = Header()
        out.header.stamp = cloud_msg.header.stamp
        out.header.frame_id = self.lidar_frame

        next_id = 0
        for cam in self.cameras.values():
            det_msg = self._find_matching_det(cam, lidar_stamp_ns)
            if det_msg is None:
                continue
            # Transform ALL LiDAR points into this camera's optical frame once.
            # Mem layout: (N,3) -> (4,N) homogeneous -> (4,N) -> (N,3)
            N = len(pts)
            ones = np.ones((N, 1), dtype=np.float32)
            pts_h = np.hstack([pts, ones]).T  # 4xN
            pts_cam = (cam.T_cam_lidar.astype(np.float32) @ pts_h).T[:, :3]

            # Only points in front of the camera (positive Z in optical frame)
            # can project to a pixel. Behind -> skip.
            z = pts_cam[:, 2]
            in_front = z > 0.1
            if not np.any(in_front):
                continue

            # Pinhole projection to pixels
            u = cam.fx * pts_cam[:, 0] / z + cam.cx
            v = cam.fy * pts_cam[:, 1] / z + cam.cy

            # Precompute mask of points usable for projection
            valid = in_front & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)

            for det in det_msg.detections:
                if not det.results:
                    continue
                cx = det.bbox.center.position.x
                cy = det.bbox.center.position.y
                w = det.bbox.size_x
                h = det.bbox.size_y
                x1 = cx - w / 2.0
                x2 = cx + w / 2.0
                y1 = cy - h / 2.0
                y2 = cy + h / 2.0

                in_box = valid & (u >= x1) & (u < x2) & (v >= y1) & (v < y2)
                n_in = int(np.count_nonzero(in_box))
                if n_in < self.min_frustum_pts:
                    continue

                frustum_pts = pts[in_box]  # back to LiDAR-frame xyz
                frustum_idx = np.where(in_box)[0]

                cluster = _dbscan_largest_nearest_cluster(
                    frustum_pts,
                    eps=self.dbscan_eps,
                    min_samples=self.dbscan_min_samples,
                    min_cluster=self.min_cluster,
                )
                if cluster is None:
                    continue

                centroid = cluster.mean(axis=0)
                r = float(np.linalg.norm(centroid[:2]))
                if r < self.min_obj_r or r > self.max_obj_r:
                    # Out-of-spec detections are silently dropped so the
                    # planner doesn't react to garbage. Logged at DEBUG.
                    self.get_logger().debug(
                        f'[{cam.name}] dropped obj at r={r:.1f}m (spec {self.min_obj_r}-{self.max_obj_r})')
                    continue

                mins = cluster.min(axis=0)
                maxs = cluster.max(axis=0)
                size = maxs - mins
                # Guard against degenerate 1D clusters (coplanar hits on a flat
                # cone face) that would produce zero-size boxes in RViz.
                size = np.maximum(size, 0.05)

                # Mark the matched points as belonging to this object
                # (we mark the full in_box set, not just the kept cluster;
                # this makes RViz colouring more legible — rejection is
                # visible but the whole box is coloured together).
                labels[frustum_idx] = next_id

                top_hyp = max(det.results, key=lambda r: r.hypothesis.score)
                d3 = Detection3D()
                d3.header = out.header
                d3.bbox = BoundingBox3D()
                d3.bbox.center = Pose()
                d3.bbox.center.position = Point(
                    x=float(centroid[0]),
                    y=float(centroid[1]),
                    z=float(centroid[2]),
                )
                d3.bbox.center.orientation.w = 1.0
                d3.bbox.size = Vector3(
                    x=float(size[0]),
                    y=float(size[1]),
                    z=float(size[2]),
                )
                h3 = ObjectHypothesisWithPose()
                h3.hypothesis.class_id = top_hyp.hypothesis.class_id
                h3.hypothesis.score = top_hyp.hypothesis.score
                h3.pose.pose = d3.bbox.center
                d3.results.append(h3)
                # id is a string in vision_msgs; stash object id + camera for
                # the viz node to colour-code and label with distance.
                d3.id = f'{cam.name}:{next_id}:{r:.2f}'

                out.detections.append(d3)
                next_id += 1

        # Publish fused objects
        self.pub_obj.publish(out)

        # Publish labeled cloud (xyz + int32 label) in LiDAR frame
        labeled_msg = self._build_labeled_cloud(cloud_msg.header, pts, labels)
        self.pub_pts.publish(labeled_msg)

    # ------------------------------------------------------------ helpers --
    def _build_labeled_cloud(
        self, header: Header, pts: np.ndarray, labels: np.ndarray
    ) -> PointCloud2:
        """Build xyz + label PointCloud2 using sensor_msgs_py.

        We use the standard create_cloud so RViz PointCloud2 display picks
        it up without any custom plugins. 'label' colouring is set in the
        RViz config via Channel Name = label on the PointCloud2 display.
        """
        fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='label', offset=12, datatype=PointField.INT32, count=1),
        ]
        data = [
            (float(p[0]), float(p[1]), float(p[2]), int(lbl))
            for p, lbl in zip(pts, labels)
        ]
        h = Header()
        h.stamp = header.stamp
        h.frame_id = self.lidar_frame
        return pc2.create_cloud(h, fields, data)


def main(args=None):
    rclpy.init(args=args)
    node = FusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
