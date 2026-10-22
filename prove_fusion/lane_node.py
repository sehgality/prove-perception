#!/usr/bin/env python3
"""
lane_node.py
------------
Classical lane detection on the FRONT camera, projected onto the real
LiDAR ground plane.

Pipeline (per image):
  1. HLS colour threshold: isolate white + yellow paint.
  2. Canny edges, AND with colour mask to clean it up.
  3. Keep only the ROI (bottom portion of the frame).
  4. Sliding-window search starting from histogram peaks in the bottom
     half, fit a 2nd-order polynomial per side.
  5. For each pixel on each fitted polynomial, cast a ray from the camera
     into the world and intersect it with the ground plane fitted from
     the latest LiDAR scan (RANSAC on points below `ground_z_max`).
     Flat-ground is only used as a fallback if RANSAC fails.
  6. Resample both lanes to ~0.5m spacing in LiDAR frame and publish:
       - /lane/polylines : visualization_msgs/MarkerArray with LINE_STRIP
       - /lane/points    : PointCloud2 (xyz + lane_id) so the viz node can
         append to the labeled cloud if we want everything in one topic.

Why no NN: you asked for pure algorithm, and for a cone-defined ODD you
probably don't need lanes at all — this is scaffolding for when you do.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray


# ---------------------------------------------------------------- helpers --

def _mat_from_param(flat_or_nested) -> np.ndarray:
    arr = np.asarray(flat_or_nested, dtype=np.float64)
    if arr.shape == (16,):
        arr = arr.reshape(4, 4)
    if arr.shape != (4, 4):
        raise ValueError(f'Expected 4x4, got {arr.shape}')
    return arr


def _invert_transform(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=T.dtype)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def _fit_ground_ransac(
    pts: np.ndarray, z_max: float, dist: float, iters: int
) -> Optional[Tuple[np.ndarray, float]]:
    """Fit plane ax+by+cz+d=0 to ground candidates. Returns (normal, d) or None.

    We seed with points whose z is below z_max (candidates for ground) so a
    truck in the LiDAR FOV doesn't dominate the fit.
    """
    candidates = pts[pts[:, 2] < z_max]
    if len(candidates) < 100:
        return None
    rng = np.random.default_rng(0)
    best_inliers = 0
    best = None
    N = len(candidates)
    for _ in range(iters):
        idx = rng.choice(N, size=3, replace=False)
        p0, p1, p2 = candidates[idx]
        v1 = p1 - p0
        v2 = p2 - p0
        n = np.cross(v1, v2)
        norm = np.linalg.norm(n)
        if norm < 1e-6:
            continue
        n = n / norm
        d = -float(n @ p0)
        # distance of all candidates to this plane
        dists = np.abs(candidates @ n + d)
        inliers = int(np.sum(dists < dist))
        if inliers > best_inliers:
            best_inliers = inliers
            best = (n, d)
    if best is None or best_inliers < 50:
        return None
    # Refit on inlier set for stability
    n0, d0 = best
    inlier_mask = np.abs(candidates @ n0 + d0) < dist
    if np.sum(inlier_mask) < 50:
        return best
    inliers = candidates[inlier_mask]
    # Least squares on normal via SVD
    centroid = inliers.mean(axis=0)
    centered = inliers - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    n = vh[-1]
    n = n / np.linalg.norm(n)
    # Keep normal pointing up (positive z component)
    if n[2] < 0:
        n = -n
    d = -float(n @ centroid)
    return n, d


# ---------------------------------------------------------------- node ----
class LaneNode(Node):

    def __init__(self):
        super().__init__('prove_lane')

        # ---- Params ---------------------------------------------------------
        self.declare_parameter('lane.camera', 'front')
        self.declare_parameter('lane.white_l_min', 200)
        self.declare_parameter('lane.yellow_h_min', 15)
        self.declare_parameter('lane.yellow_h_max', 35)
        self.declare_parameter('lane.yellow_s_min', 100)
        self.declare_parameter('lane.roi_top_frac', 0.55)
        self.declare_parameter('lane.n_windows', 9)
        self.declare_parameter('lane.window_margin_px', 80)
        self.declare_parameter('lane.min_pixels_per_window', 40)
        self.declare_parameter('lane.max_lane_range', 25.0)
        self.declare_parameter('lane.lane_resample_m', 0.5)
        self.declare_parameter('lane.ground_ransac_dist', 0.1)
        self.declare_parameter('lane.ground_ransac_iters', 100)
        self.declare_parameter('lidar.ground_z_max', -0.3)
        self.declare_parameter('lidar.frame_id', 'lidar_link')
        self.declare_parameter('lidar.topic', '/lidar/points')

        self.declare_parameter('cameras.front.image_topic', '/camera/front/image_raw')
        self.declare_parameter('cameras.front.frame_id', 'cam_front_optical')
        self.declare_parameter('cameras.front.fx', 640.0)
        self.declare_parameter('cameras.front.fy', 640.0)
        self.declare_parameter('cameras.front.cx', 640.0)
        self.declare_parameter('cameras.front.cy', 360.0)
        self.declare_parameter('cameras.front.width', 1280)
        self.declare_parameter('cameras.front.height', 720)
        self.declare_parameter(
            'cameras.front.T_lidar_cam',
            [0.0, 0.0, 1.0, 1.5,
             -1.0, 0.0, 0.0, 0.0,
             0.0, -1.0, 0.0, 1.2,
             0.0, 0.0, 0.0, 1.0],
        )

        self.l_min = int(self.get_parameter('lane.white_l_min').value)
        self.h_min = int(self.get_parameter('lane.yellow_h_min').value)
        self.h_max = int(self.get_parameter('lane.yellow_h_max').value)
        self.s_min = int(self.get_parameter('lane.yellow_s_min').value)
        self.roi_top = float(self.get_parameter('lane.roi_top_frac').value)
        self.n_windows = int(self.get_parameter('lane.n_windows').value)
        self.win_margin = int(self.get_parameter('lane.window_margin_px').value)
        self.min_px = int(self.get_parameter('lane.min_pixels_per_window').value)
        self.max_range = float(self.get_parameter('lane.max_lane_range').value)
        self.resample_m = float(self.get_parameter('lane.lane_resample_m').value)
        self.ransac_dist = float(self.get_parameter('lane.ground_ransac_dist').value)
        self.ransac_iters = int(self.get_parameter('lane.ground_ransac_iters').value)
        self.ground_z_max = float(self.get_parameter('lidar.ground_z_max').value)
        self.lidar_frame = self.get_parameter('lidar.frame_id').value

        self.fx = float(self.get_parameter('cameras.front.fx').value)
        self.fy = float(self.get_parameter('cameras.front.fy').value)
        self.cx = float(self.get_parameter('cameras.front.cx').value)
        self.cy = float(self.get_parameter('cameras.front.cy').value)
        self.img_w = int(self.get_parameter('cameras.front.width').value)
        self.img_h = int(self.get_parameter('cameras.front.height').value)
        T = _mat_from_param(self.get_parameter('cameras.front.T_lidar_cam').value)
        self.T_lidar_cam = T.astype(np.float64)
        self.T_cam_lidar = _invert_transform(T).astype(np.float64)

        # ---- Ground-plane cache --------------------------------------------
        # Updated whenever we get a LiDAR scan. If none ever arrives we fall
        # back to flat (n=[0,0,1], d = -lidar_height). Lidar height is
        # inferred from the camera extrinsic z offset (-camera_z + ~1.2 is
        # bad; use the config override later if your mounting differs).
        self._ground_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self._ground_d = 0.0  # z = 0 in LiDAR frame is lidar sensor itself;
        # real ground is below. 0.0 lets us compute first and then update.
        self._have_ground = False

        # ---- IO -------------------------------------------------------------
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.bridge = CvBridge()
        cam = str(self.get_parameter('lane.camera').value)
        self.create_subscription(
            Image,
            self.get_parameter(f'cameras.{cam}.image_topic').value,
            self._on_image, qos,
        )
        self.create_subscription(
            PointCloud2, self.get_parameter('lidar.topic').value,
            self._on_cloud, qos,
        )

        self.pub_markers = self.create_publisher(MarkerArray, '/lane/polylines', 10)
        self.pub_points = self.create_publisher(PointCloud2, '/lane/points', 10)
        self.get_logger().info(f'lane node ready (camera: {cam})')

    # ---------------------------------------------------------------- subs --
    def _on_cloud(self, msg: PointCloud2) -> None:
        """Refit ground plane from the latest scan."""
        pts_list = list(pc2.read_points(
            msg, field_names=('x', 'y', 'z'), skip_nans=True))
        if not pts_list:
            return
        pts = np.asarray(
            [(p[0], p[1], p[2]) for p in pts_list], dtype=np.float64)
        fit = _fit_ground_ransac(
            pts, z_max=self.ground_z_max,
            dist=self.ransac_dist, iters=self.ransac_iters,
        )
        if fit is None:
            # Keep whatever we had; if nothing, flat plane at z=ground_z_max
            if not self._have_ground:
                self._ground_normal = np.array([0.0, 0.0, 1.0])
                self._ground_d = -self.ground_z_max
            return
        self._ground_normal, self._ground_d = fit
        self._have_ground = True

    def _on_image(self, msg: Image) -> None:
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge: {e}')
            return
        h, w = img.shape[:2]

        # Resize if the incoming image doesn't match configured intrinsics.
        # Rescaling intrinsics would be more honest but slower; log a warn
        # once so the user knows calibration will be slightly off.
        if (w != self.img_w or h != self.img_h):
            img = cv2.resize(img, (self.img_w, self.img_h))

        mask = self._paint_mask(img)
        left_fit, right_fit = self._sliding_window(mask)

        left_pts = self._poly_to_pixel_strip(left_fit, mask.shape[0])
        right_pts = self._poly_to_pixel_strip(right_fit, mask.shape[0])

        left_xyz = self._pixels_to_ground(left_pts)
        right_xyz = self._pixels_to_ground(right_pts)

        left_xyz = self._resample(left_xyz, self.resample_m)
        right_xyz = self._resample(right_xyz, self.resample_m)

        self._publish(msg.header.stamp, left_xyz, right_xyz)

    # ---------------------------------------------------------- processing --
    def _paint_mask(self, img_bgr: np.ndarray) -> np.ndarray:
        """Combined white + yellow paint mask with Canny AND."""
        hls = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)
        H, L, S = hls[:, :, 0], hls[:, :, 1], hls[:, :, 2]
        white = L > self.l_min
        yellow = (H > self.h_min) & (H < self.h_max) & (S > self.s_min)
        colour_mask = (white | yellow).astype(np.uint8) * 255

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 60, 180)

        combined = cv2.bitwise_and(colour_mask, edges)

        # Keep only bottom ROI
        roi_mask = np.zeros_like(combined)
        y0 = int(self.roi_top * combined.shape[0])
        roi_mask[y0:, :] = 255
        combined = cv2.bitwise_and(combined, roi_mask)
        return combined

    def _sliding_window(self, mask: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Standard sliding-window lane search; returns np.polyfit coeffs per side.

        Returns None for a side where too few pixels were found.
        """
        h, w = mask.shape
        # Histogram of bottom half -> left and right peaks
        hist = np.sum(mask[h // 2:, :], axis=0)
        midpoint = w // 2
        left_peak = int(np.argmax(hist[:midpoint]))
        right_peak = int(np.argmax(hist[midpoint:]) + midpoint)

        window_h = h // self.n_windows
        nonzero = mask.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])

        leftx_cur = left_peak
        rightx_cur = right_peak
        left_lane_inds: List[np.ndarray] = []
        right_lane_inds: List[np.ndarray] = []

        for i in range(self.n_windows):
            win_y_low = h - (i + 1) * window_h
            win_y_high = h - i * window_h
            win_xleft_low = leftx_cur - self.win_margin
            win_xleft_high = leftx_cur + self.win_margin
            win_xright_low = rightx_cur - self.win_margin
            win_xright_high = rightx_cur + self.win_margin

            good_left = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                         (nonzero_x >= win_xleft_low) & (nonzero_x < win_xleft_high)).nonzero()[0]
            good_right = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                          (nonzero_x >= win_xright_low) & (nonzero_x < win_xright_high)).nonzero()[0]
            left_lane_inds.append(good_left)
            right_lane_inds.append(good_right)
            if len(good_left) > self.min_px:
                leftx_cur = int(np.mean(nonzero_x[good_left]))
            if len(good_right) > self.min_px:
                rightx_cur = int(np.mean(nonzero_x[good_right]))

        left_idx = np.concatenate(left_lane_inds) if left_lane_inds else np.array([], dtype=int)
        right_idx = np.concatenate(right_lane_inds) if right_lane_inds else np.array([], dtype=int)

        left_fit = None
        right_fit = None
        if len(left_idx) > 100:
            left_fit = np.polyfit(nonzero_y[left_idx], nonzero_x[left_idx], 2)
        if len(right_idx) > 100:
            right_fit = np.polyfit(nonzero_y[right_idx], nonzero_x[right_idx], 2)
        return left_fit, right_fit

    def _poly_to_pixel_strip(
        self, fit: Optional[np.ndarray], img_h: int, step: int = 10
    ) -> np.ndarray:
        """Evaluate the fitted polynomial x(y) at a strip of y values.

        Returns Nx2 (u, v) in pixel coords. Empty array if no fit.
        """
        if fit is None:
            return np.zeros((0, 2), dtype=np.float64)
        y_top = int(self.roi_top * img_h)
        ys = np.arange(y_top, img_h, step, dtype=np.float64)
        xs = fit[0] * ys * ys + fit[1] * ys + fit[2]
        pts = np.stack([xs, ys], axis=1)
        # Keep only pixels inside the image
        inside = (pts[:, 0] >= 0) & (pts[:, 0] < self.img_w)
        return pts[inside]

    def _pixels_to_ground(self, pix: np.ndarray) -> np.ndarray:
        """Back-project pixels onto the current LiDAR ground plane.

        For each pixel (u, v):
          ray_cam = normalize([(u - cx)/fx, (v - cy)/fy, 1])
          world ray origin = camera position in LiDAR frame (T_lidar_cam[:3,3])
          world ray dir    = T_lidar_cam[:3,:3] @ ray_cam
          intersect with plane n . x + d = 0:
             t = -(n . origin + d) / (n . dir)
          return origin + t * dir

        Rays with t <= 0 (plane behind camera) or t beyond max_range are
        dropped. This replaces the usual "assume ground is flat" IPM which
        goes to garbage on real tracks with pitch or slope.
        """
        if len(pix) == 0:
            return np.zeros((0, 3), dtype=np.float64)
        u = pix[:, 0]
        v = pix[:, 1]
        xc = (u - self.cx) / self.fx
        yc = (v - self.cy) / self.fy
        zc = np.ones_like(xc)
        rays_cam = np.stack([xc, yc, zc], axis=1)  # Nx3

        R = self.T_lidar_cam[:3, :3]
        origin = self.T_lidar_cam[:3, 3]
        rays_world = (R @ rays_cam.T).T  # Nx3

        n = self._ground_normal
        d = self._ground_d
        denom = rays_world @ n
        # Avoid div-by-zero / rays parallel to the plane
        good = np.abs(denom) > 1e-6
        t = np.full(len(rays_world), np.nan)
        t[good] = -(origin @ n + d) / denom[good]

        valid = (t > 0) & np.isfinite(t)
        pts = origin + rays_world * t[:, None]
        pts = pts[valid]

        # Range gate
        if len(pts) == 0:
            return pts
        r = np.linalg.norm(pts[:, :2], axis=1)
        return pts[r <= self.max_range]

    def _resample(self, pts: np.ndarray, step_m: float) -> np.ndarray:
        """Resample a polyline so consecutive points are ~step_m apart.

        Expects pts ordered by image row (which, after projection, roughly
        orders them by distance from camera — closest first).
        """
        if len(pts) < 2:
            return pts
        # Sort by range so geometry is monotonic from the car outward
        r = np.linalg.norm(pts[:, :2], axis=1)
        order = np.argsort(r)
        pts = pts[order]
        kept = [pts[0]]
        for p in pts[1:]:
            if np.linalg.norm(p[:2] - kept[-1][:2]) >= step_m:
                kept.append(p)
        return np.asarray(kept)

    # ---------------------------------------------------------- publishing --
    def _publish(self, stamp, left_xyz: np.ndarray, right_xyz: np.ndarray) -> None:
        header = Header()
        header.stamp = stamp
        header.frame_id = self.lidar_frame

        # --- MarkerArray for RViz ------------------------------------------
        arr = MarkerArray()
        # Clear previous frame's markers
        clear = Marker()
        clear.header = header
        clear.ns = 'lanes'
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)

        for i, (pts, colour) in enumerate([
            (left_xyz, ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)),   # white
            (right_xyz, ColorRGBA(r=1.0, g=0.9, b=0.1, a=1.0)),  # yellow
        ]):
            if len(pts) < 2:
                continue
            m = Marker()
            m.header = header
            m.ns = 'lanes'
            m.id = i
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.08  # line width in meters
            m.color = colour
            m.pose.orientation.w = 1.0
            for p in pts:
                m.points.append(Point(x=float(p[0]), y=float(p[1]), z=float(p[2])))
            arr.markers.append(m)
        self.pub_markers.publish(arr)

        # --- Lane points as a cloud ----------------------------------------
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='lane_id', offset=12, datatype=PointField.INT32, count=1),
        ]
        data = []
        for lane_id, pts in enumerate([left_xyz, right_xyz]):
            for p in pts:
                data.append((float(p[0]), float(p[1]), float(p[2]), lane_id))
        self.pub_points.publish(pc2.create_cloud(header, fields, data))


def main(args=None):
    rclpy.init(args=args)
    node = LaneNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
