#!/usr/bin/env python3
"""
viz_node.py
-----------
Turns fused Detection3DArray messages into a MarkerArray that RViz2
renders nicely: cube per object coloured by class, line border, and a
TEXT_VIEW_FACING marker showing "<class> <range>m" above each box.

Kept separate from the fusion node for two reasons:
  - Markers are chatty; sending them from a dedicated QoS keeps the
    fusion node's publisher queues from backing up when RViz subscribes.
  - You can run fusion headlessly on the Jetson and only start the viz
    + RViz on your monitor when you want to actually look at it.
"""

from __future__ import annotations

import hashlib
from typing import Dict

import rclpy
from geometry_msgs.msg import Point, Vector3
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import ColorRGBA, Header
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray


# A small, high-contrast palette for class colours. Hash maps class name
# into this list so identical class ids get the same colour across frames
# and across runs.
_PALETTE = [
    (0.95, 0.26, 0.21),  # red
    (0.91, 0.59, 0.00),  # orange
    (0.95, 0.76, 0.05),  # yellow
    (0.30, 0.69, 0.31),  # green
    (0.00, 0.74, 0.83),  # cyan
    (0.25, 0.32, 0.71),  # indigo
    (0.61, 0.15, 0.69),  # purple
    (0.80, 0.80, 0.80),  # gray
]


def _color_for_class(class_id: str, alpha: float = 0.6) -> ColorRGBA:
    h = int(hashlib.md5(class_id.encode('utf-8')).hexdigest(), 16)
    r, g, b = _PALETTE[h % len(_PALETTE)]
    return ColorRGBA(r=r, g=g, b=b, a=alpha)


class VizNode(Node):

    def __init__(self):
        super().__init__('prove_viz')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.create_subscription(
            Detection3DArray, '/fusion/objects', self._on_objects, qos)
        self.pub = self.create_publisher(MarkerArray, '/fusion/markers', qos)
        self.get_logger().info('viz node ready')

    def _on_objects(self, msg: Detection3DArray) -> None:
        arr = MarkerArray()
        header = msg.header

        # Clear previous frame markers in this namespace so stale boxes
        # don't linger (RViz holds markers until TTL even on new frames).
        clear = Marker()
        clear.header = header
        clear.ns = 'objects'
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        clear2 = Marker()
        clear2.header = header
        clear2.ns = 'labels'
        clear2.action = Marker.DELETEALL
        arr.markers.append(clear2)

        for i, det in enumerate(msg.detections):
            if not det.results:
                continue
            top = det.results[0]
            cls = top.hypothesis.class_id
            score = top.hypothesis.score

            # cube
            cube = Marker()
            cube.header = header
            cube.ns = 'objects'
            cube.id = i
            cube.type = Marker.CUBE
            cube.action = Marker.ADD
            cube.pose = det.bbox.center
            cube.scale = det.bbox.size
            cube.color = _color_for_class(cls, alpha=0.8)
            cube.lifetime.sec = 0  # cleared next frame via DELETEALL
            arr.markers.append(cube)

            # text label just above the cube
            txt = Marker()
            txt.header = header
            txt.ns = 'labels'
            txt.id = i
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            txt.pose.position = Point(
                x=det.bbox.center.position.x,
                y=det.bbox.center.position.y,
                z=det.bbox.center.position.z + float(det.bbox.size.z) / 2.0 + 0.3,
            )
            txt.pose.orientation.w = 1.0
            txt.scale = Vector3(x=0.0, y=0.0, z=1.2)  # z = text height
            txt.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)

            # det.id is "cam:obj_id:range_m" from the fusion node
            r_str = det.id.split(':')[-1] if ':' in det.id else '?'
            txt.text = f'{cls} {score:.2f} | {r_str} m'
            arr.markers.append(txt)

        self.pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = VizNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
