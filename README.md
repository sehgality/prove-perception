# PROVE Perception

Camera + LiDAR fusion stack for the PROVE autonomous EV. Runs on an NVIDIA Jetson Orin Nano, consumes 3 cameras + 1 Ouster OS1 LiDAR over ROS2, and outputs 3D object boxes with class labels and LiDAR-measured distances, a labeled point cloud, and lane line polylines projected onto the ground plane. Visualized in RViz2 running on an operator laptop.

## Architecture

Isaac Sim (or the real car) publishes raw camera + LiDAR streams over ROS2 to the Jetson. The Jetson runs the detector, fusion, lane detection, and viz nodes headlessly. An operator laptop subscribes to the fused output topics and renders them in RViz2. Keeps compute on the edge, visualization on the operator machine.

## Repo layout

- config/sensors.yaml - ALL calibration and tuning
- config/fastdds_discovery_client.xml - network discovery fallback
- launch/prove_fusion.launch.py - one-command startup
- rviz/prove_fusion.rviz - preloaded RViz view
- prove_fusion/detector_node.py - YOLOv8 on 3 cameras
- prove_fusion/fusion_node.py - camera 2D box plus LiDAR to 3D box with distance
- prove_fusion/lane_node.py - classical lane detection, ground-plane projection
- prove_fusion/viz_node.py - 3D cube markers plus distance labels for RViz
- package.xml, setup.py, setup.cfg - ROS2 package metadata

## What is working

- End-to-end pipeline: Isaac Sim to ROS2 to Jetson to 3D fused boxes to RViz
- Camera-LiDAR frustum fusion with DBSCAN clustering per 2D detection
- Classical lane detection (HLS color plus Canny plus sliding window plus ground-plane RANSAC)
- Chase-cam Orbit view in RViz centered on the vehicle LiDAR frame

## What is not yet calibrated or tuned

- Camera-LiDAR extrinsics (T_lidar_cam) are placeholder guesses, mechanical team provides real values
- YOLOv8n is COCO-trained, detects cars, people, stop signs, etc. Does NOT detect cones. See feature/cone-detector
- LiDAR runs at about 5 Hz because of WiFi bandwidth limits, use wired ethernet for better rates
- RViz on the Jetson starves the detector of DDS buffer, always use rviz:=false on the Jetson side

## Install on Jetson Orin Nano (JetPack 6, Ubuntu 22.04)

Run these commands:

    sudo apt install -y ros-humble-desktop ros-humble-vision-msgs ros-humble-cv-bridge python3-colcon-common-extensions
    pip3 install ultralytics opencv-python 'numpy<2' scikit-learn
    mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
    git clone https://github.com/sehgality/prove-perception.git prove_fusion
    cd ~/ros2_ws
    colcon build --symlink-install --packages-select prove_fusion
    echo 'source ~/ros2_ws/install/setup.bash' >> ~/.bashrc
    source ~/.bashrc

Also bump kernel buffers so large sensor messages do not drop. Add these to /etc/sysctl.conf:

    net.core.rmem_max=2147483647
    net.core.rmem_default=2147483647
    net.ipv4.ipfrag_time=3
    net.ipv4.ipfrag_high_thresh=134217728

Then run: sudo sysctl -p

## Install on operator laptop (RViz display only)

WSL2 Ubuntu 22.04 or native Ubuntu with ROS2 Humble:

    sudo apt install -y ros-humble-desktop ros-humble-vision-msgs ros-humble-cv-bridge
    scp aidan@<jetson-ip>:~/ros2_ws/src/prove_fusion/rviz/prove_fusion.rviz ~/

## Run

On the Jetson (headless, over SSH):

    ros2 launch prove_fusion prove_fusion.launch.py rviz:=false

Watch for lines like `[detector] front: X fps in, Y det/s`. Any fps in greater than 0 means frames are flowing.

On the laptop (RViz display):

    rviz2 -d ~/prove_fusion.rviz

## ROS2 topic graph

Inputs expected from Isaac Sim or the real car:
- /sensors/camera/{front,left,right}/image_raw   sensor_msgs/Image
- /sensors/camera/{front,left,right}/camera_info sensor_msgs/CameraInfo
- /sensors/lidar/points                          sensor_msgs/PointCloud2

Outputs published by this stack:
- /detections/{front,left,right}  vision_msgs/Detection2DArray
- /fusion/objects                 vision_msgs/Detection3DArray
- /fusion/points_labeled          sensor_msgs/PointCloud2 (plus label field)
- /fusion/markers                 visualization_msgs/MarkerArray
- /lane/polylines                 visualization_msgs/MarkerArray

## Configuration

Everything lives in config/sensors.yaml.

Key params:
- detector.weights - YOLO model path (.pt or .engine)
- detector.conf_threshold - min detection confidence (0.05 trigger-happy, 0.5 conservative)
- cameras.*.T_lidar_cam - 4x4 transform of each camera in the LiDAR frame
- fusion.sync_slop_sec - max time delta between camera plus lidar for a match (default 50 ms)
- fusion.max_object_range - kill distant garbage (default 40 m, matches ODD spec)
- lane.* - classical lane detection tuning

## Branching strategy

- main - always works, deployable to Jetson
- dev - active development, PRs land here first
- feature/<name> - individual work, branched from dev

Suggested features (per PDR feedback):
- feature/cone-detector - retrain YOLO on cone dataset (FSOCO is free)
- feature/tensorrt-export - export engine for 3-5x inference speedup
- feature/motion-deskew - compensate LiDAR scan skew while turning
- feature/eskf-fault-gates - GPS/RTK loss behavior
- feature/camera-only-baseline - fallback perception without LiDAR
- feature/sensor-noise-injection - stress-test fusion with realistic noise

## For camera vision contributors

Primary file: prove_fusion/detector_node.py

To swap in a new model (e.g., cone-trained YOLO):
1. Train it, save as .pt or export as .engine
2. Drop the file on the Jetson (model files are gitignored, do not commit them)
3. In config/sensors.yaml, point detector.weights at your file path
4. Rebuild (colcon build --symlink-install --packages-select prove_fusion) and relaunch

Any new detection nodes should publish vision_msgs/Detection2DArray to /detections/{camera_name}. fusion_node.py picks them up automatically.

## Known issues

- RViz on the Jetson starves the detector. RViz hogs the DDS buffer. Always use rviz:=false on the Jetson. Run RViz on the laptop instead.
- LiDAR density is capped by WiFi. Jetson over campus WiFi cannot sustain 3x cams at high res. Use wired ethernet, or 416x416 resolution for cameras.
- Fusion extrinsics are default placeholders. All 3D boxes will be offset until the real T_lidar_cam matrices are measured and dropped in sensors.yaml.
- YOLOv8n detects COCO classes only (car, person, traffic light, stop sign, etc). Does NOT detect cones. Train a cone-specific model.
- TensorRT export is broken on current JetPack 6 plus Ultralytics. Ultralytics calls torch.onnx.export(dynamo=...) which errors on the Jetson bundled PyTorch 2.4. Workaround: export ONNX elsewhere, use trtexec on the Jetson to build the engine.

## PDR follow-ups baked into the roadmap

- Define minimum detection distance spec (fusion.min_object_range pending mechanical dead-zone measurements)
- Motion deskew for LiDAR scans while turning
- ESKF fault thresholds plus GPS-denied behavior
- Camera-only baseline fallback
- Write-up on why LiDAR beats camera-only for our ODD
- Real-data logging from manual drives (highest priority per William)

## Maintainers

- Aidan Sehgal - perception stack, fusion, Jetson deployment
