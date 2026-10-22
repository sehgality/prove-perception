#!/usr/bin/env python3
"""
detector_node.py
----------------
Per-camera 2D object detector.

Subscribes to one or more camera image topics and publishes a
vision_msgs/Detection2DArray for each. The fusion node consumes those
arrays, so this node's job is just: image in, bounding boxes out.

Why one node that handles all cameras instead of three nodes:
  - A single YOLO/TRT engine load is heavy on an 8GB Orin Nano; sharing
    saves ~500MB and lets the model stay warm between frames.
  - Letting one process see all three cameras makes it trivial to add
    per-camera queue control / frame drop later.

Hard rule for a Jetson: at 1080p30 x 3 cameras you'll melt the Orin Nano.
Either downsample incoming images (set Isaac Sim to 640x480), publish
compressed, or add a frame skipper here. We default to processing every
Nth frame per camera (frame_skip param) and resizing the input internally.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import (
    BoundingBox2D,
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)

# Lazy-imported in __init__ so this module stays importable even without
# ultralytics installed (useful for unit tests on CI without GPU).
_YOLO = None


@dataclass
class CameraCfg:
    name: str
    image_topic: str
    frame_id: str


class DetectorNode(Node):

    def __init__(self):
        super().__init__('prove_detector')

        # ---- Params ---------------------------------------------------------
        self.declare_parameter('weights', 'yolov8n.pt')
        self.declare_parameter('conf_threshold', 0.35)
        self.declare_parameter('iou_threshold', 0.5)
        self.declare_parameter('input_size', 640)
        self.declare_parameter('device', 'cuda:0')
        self.declare_parameter('frame_skip', 1)  # 1 = process every frame
        self.declare_parameter('use_tensorrt', False)

        # Camera list is parsed from flat params to keep the YAML simple:
        # cameras.front.image_topic, cameras.front.frame_id, ...
        self.declare_parameter('cameras.front.image_topic', '/camera/front/image_raw')
        self.declare_parameter('cameras.front.frame_id', 'cam_front_optical')
        self.declare_parameter('cameras.left.image_topic', '/camera/left/image_raw')
        self.declare_parameter('cameras.left.frame_id', 'cam_left_optical')
        self.declare_parameter('cameras.right.image_topic', '/camera/right/image_raw')
        self.declare_parameter('cameras.right.frame_id', 'cam_right_optical')

        weights = self.get_parameter('weights').value
        self.conf = float(self.get_parameter('conf_threshold').value)
        self.iou = float(self.get_parameter('iou_threshold').value)
        self.imgsz = int(self.get_parameter('input_size').value)
        self.device = self.get_parameter('device').value
        self.frame_skip = max(1, int(self.get_parameter('frame_skip').value))
        self.use_trt = bool(self.get_parameter('use_tensorrt').value)

        # ---- Model ----------------------------------------------------------
        global _YOLO
        if _YOLO is None:
            # Imported here so the ROS node fails loudly only when actually run,
            # not when imported for testing or docs generation.
            from ultralytics import YOLO as _Y  # type: ignore
            _YOLO = _Y
        self.get_logger().info(f'Loading detector weights: {weights}')
        self.model = _YOLO(weights)
        if self.use_trt:
            # ultralytics will transparently use .engine if the weights path
            # points to one, so you'd typically pass weights=yolov8n.engine
            # from the YAML. This flag is for logging only.
            self.get_logger().info('TensorRT backend assumed (weights path should be .engine)')

        # Warm the GPU with a dummy inference so the first real frame isn't 2s
        try:
            self.model.predict(
                np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8),
                verbose=False, device=self.device, imgsz=self.imgsz,
            )
            self.get_logger().info('Model warmed up')
        except Exception as e:
            # Non-fatal; the first inference will just be slow
            self.get_logger().warn(f'Warmup failed (non-fatal): {e}')

        # ---- IO -------------------------------------------------------------
        self.bridge = CvBridge()
        # BEST_EFFORT + small depth matches Isaac Sim's camera publishers; a
        # RELIABLE sub against a BEST_EFFORT pub will silently receive nothing.
        self.qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.cameras: Dict[str, CameraCfg] = {}
        self.frame_count: Dict[str, int] = {}
        self.publishers_: Dict[str, rclpy.publisher.Publisher] = {}

        for name in ('front', 'left', 'right'):
            topic = self.get_parameter(f'cameras.{name}.image_topic').value
            frame = self.get_parameter(f'cameras.{name}.frame_id').value
            cfg = CameraCfg(name=name, image_topic=topic, frame_id=frame)
            self.cameras[name] = cfg
            self.frame_count[name] = 0

            det_topic = f'/detections/{name}'
            self.publishers_[name] = self.create_publisher(
                Detection2DArray, det_topic, self.qos)

            # Bind name into the callback so one handler serves all cams
            self.create_subscription(
                Image, topic,
                lambda msg, n=name: self._on_image(n, msg),
                self.qos,
            )
            self.get_logger().info(f'[{name}] subscribed {topic} -> {det_topic}')

        # Running stats for a periodic diagnostic log line
        self._last_log = time.time()
        self._det_counts: Dict[str, int] = {n: 0 for n in self.cameras}
        self.create_timer(5.0, self._log_stats)

    # ------------------------------------------------------------------ cb --
    def _on_image(self, cam_name: str, msg: Image) -> None:
        """Handle one incoming image frame."""
        self.frame_count[cam_name] += 1
        if self.frame_count[cam_name] % self.frame_skip != 0:
            return

        try:
            # bgr8 so OpenCV / YOLO both work natively
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'[{cam_name}] cv_bridge failed: {e}')
            return

        try:
            results = self.model.predict(
                cv_img,
                verbose=False,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                device=self.device,
            )
        except Exception as e:
            self.get_logger().error(f'[{cam_name}] inference failed: {e}')
            return

        det_arr = self._results_to_msg(results, msg, cam_name)
        self.publishers_[cam_name].publish(det_arr)
        self._det_counts[cam_name] += len(det_arr.detections)

    def _results_to_msg(self, results, img_msg: Image, cam_name: str) -> Detection2DArray:
        """Convert ultralytics YOLO result -> vision_msgs/Detection2DArray.

        Timestamp comes from the IMAGE, not the current clock, so the fusion
        node can time-match this detection against a LiDAR scan.
        """
        out = Detection2DArray()
        out.header = img_msg.header  # keeps original stamp + camera frame_id

        if not results:
            return out
        r = results[0]  # ultralytics returns a list with one entry per image
        if r.boxes is None or len(r.boxes) == 0:
            return out

        # xyxy in pixel coords, cls as int tensor, conf as float tensor
        boxes = r.boxes.xyxy.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        names = r.names  # dict id -> label

        for (x1, y1, x2, y2), c, p in zip(boxes, clss, confs):
            d = Detection2D()
            d.header = out.header

            bb = BoundingBox2D()
            # vision_msgs convention: center + size, not corners
            bb.center.position.x = float((x1 + x2) / 2.0)
            bb.center.position.y = float((y1 + y2) / 2.0)
            bb.center.theta = 0.0
            bb.size_x = float(x2 - x1)
            bb.size_y = float(y2 - y1)
            d.bbox = bb

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(names.get(int(c), str(c)))
            hyp.hypothesis.score = float(p)
            d.results.append(hyp)

            out.detections.append(d)
        return out

    def _log_stats(self) -> None:
        now = time.time()
        dt = now - self._last_log
        if dt <= 0:
            return
        parts = []
        for name in self.cameras:
            fps_in = self.frame_count[name] / dt
            det_per_s = self._det_counts[name] / dt
            parts.append(f'{name}: {fps_in:.1f} fps in, {det_per_s:.1f} det/s')
            self.frame_count[name] = 0
            self._det_counts[name] = 0
        self._last_log = now
        self.get_logger().info('[detector] ' + ' | '.join(parts))


def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
