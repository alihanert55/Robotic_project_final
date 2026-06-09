#!/usr/bin/env python3
"""
odometry_node.py – Wheel Encoder + Monocular ORB SLAM Fusion
=============================================================
Two-source localization following the LearnOpenCV monocular SLAM
pattern, adapted for Duckiebot differential drive.

  Source 1 – Wheel encoders (always runs):
      Differential-drive kinematics, same as Assignment 2.

  Source 2 – Monocular ORB Visual Odometry:
      ORB features → knnMatch → Lowe's ratio test →
      Essential matrix (RANSAC) → recoverPose → R, t.
      Translation is unit-vector only (monocular has no scale),
      so we scale it using wheel encoder delta_s.

  Fusion:
      heading  = (1 - alpha) * encoder_theta + alpha * vo_theta
      position = encoder_position + scaled VO correction
      When VO fails (few matches, blur) → pure wheel fallback.

Published:  /odometry (nav_msgs/Odometry)
"""

import math
import os

import cv2
import numpy as np
import rospy
import yaml
import tf.transformations
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3
from sensor_msgs.msg import CompressedImage, CameraInfo
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelEncoderStamped


def load_config():
    """Load config.yaml from same folder as this script."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)


class OdometryNode(DTROS):
    """Wheel encoder + ORB monocular SLAM odometry node.

    Same DTROS pattern as Assignment 2 but ArUco replaced with
    ORB visual odometry following LearnOpenCV monocular SLAM.
    """

    def __init__(self, node_name):
        super(OdometryNode, self).__init__(
            node_name=node_name, node_type=NodeType.LOCALIZATION)

        self.cfg = load_config()
        self._vehicle_name = os.environ.get(
            "VEHICLE_NAME", self.cfg["vehicle_name"])

        # ── robot params ─────────────────────────────────────
        self.wheel_base = self.cfg["robot"]["wheel_base"]
        self.ticks_per_meter = 716.0

        # ── pose state (start from config) ───────────────────
        self.x = self.cfg["start"]["x"]
        self.y = self.cfg["start"]["y"]
        self.theta = 0.0

        # ── encoder state ────────────────────────────────────
        self.start_left_ticks = None
        self.start_right_ticks = None
        self.left_ticks = 0
        self.right_ticks = 0
        self.last_left_ticks = 0
        self.last_right_ticks = 0
        self.last_time = rospy.Time.now()

        # ── ORB visual odometry ──────────────────────────────
        self.orb = cv2.ORB_create(nfeatures=200)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)  # knnMatch needs crossCheck=False

        self.prev_kp = None
        self.prev_des = None

        # Camera intrinsics (loaded dynamically via camera_info_cb)
        self.K = None
        self.D = None
        self.mapx = None
        self.mapy = None

        # VO results (consumed each update cycle)
        self.vo_delta_theta = 0.0   # yaw change from VO
        self.vo_tx = 0.0            # translation direction x
        self.vo_ty = 0.0            # translation direction y
        self.vo_valid = False

        # fusion weights
        self.alpha = 0.4            # VO trust (0 = pure encoder, 1 = pure VO)
        self.min_matches = 75       # minimum inliers for valid VO
        self.lowe_ratio = 0.75      # Lowe's ratio test threshold

        # ── subscribers ──────────────────────────────────────
        rospy.Subscriber(
            f"/{self._vehicle_name}/left_wheel_encoder_node/tick",
            WheelEncoderStamped, self.cb_left_encoder)
        rospy.Subscriber(
            f"/{self._vehicle_name}/right_wheel_encoder_node/tick",
            WheelEncoderStamped, self.cb_right_encoder)
        rospy.Subscriber(
            f"/{self._vehicle_name}/camera_node/image/compressed",
            CompressedImage, self.cb_image,
            queue_size=1, buff_size=2**24)
        self.info_sub = rospy.Subscriber(
            f"/{self._vehicle_name}/camera_node/camera_info",
            CameraInfo, self.camera_info_cb)

        # ── publisher ────────────────────────────────────────
        self.pub_odom = rospy.Publisher("/odometry", Odometry, queue_size=10)

        # ── timer (10 Hz, same as Assignment 2) ──────────────
        self.timer = rospy.Timer(rospy.Duration(0.1), self.update_odometry)
        rospy.loginfo("[%s] Wheel + ORB monocular SLAM odometry started.", node_name)

    # ── encoder callbacks ────────────────────────────────────

    def cb_left_encoder(self, msg):
        """Store latest left wheel tick count."""
        if self.start_left_ticks is None:
            self.start_left_ticks = msg.data
        self.left_ticks = msg.data - self.start_left_ticks

    def cb_right_encoder(self, msg):
        """Store latest right wheel tick count."""
        if self.start_right_ticks is None:
            self.start_right_ticks = msg.data
        self.right_ticks = msg.data - self.start_right_ticks

    def camera_info_cb(self, msg):
        if self.K is None:
            self.K = np.array(msg.K, dtype=np.float64).reshape(3, 3)
            self.D = np.array(msg.D, dtype=np.float64)
            
            # Compute the undistortion map once
            self.mapx, self.mapy = cv2.initUndistortRectifyMap(
                self.K, self.D, None, self.K, (msg.width, msg.height), cv2.CV_32FC1)
            rospy.loginfo("[INFO] Calibration matrices and remap coordinates successfully initialized!")
            
            # Unsubscribe since calibration data is static
            self.info_sub.unregister()

    # ── ORB visual odometry ──────────────────────────────────

    def cb_image(self, msg):
        """Process camera frame: ORB detect → match → Essential → R, t.

        Uses Lowe's ratio test (same as LearnOpenCV / SLAMPy pattern)
        to filter matches before estimating the Essential matrix.
        """
        # Wait until camera calibration is initialized
        if self.K is None or self.mapx is None or self.mapy is None:
            return

        # decode compressed image to grayscale
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        if frame is None:
            return

        # Undistort the frame using the precomputed map
        frame = cv2.remap(frame, self.mapx, self.mapy, cv2.INTER_LINEAR)

        # detect ORB keypoints + descriptors
        kp, des = self.orb.detectAndCompute(frame, None)
        if des is None:
            self.prev_kp, self.prev_des = kp, des
            self.vo_valid = False
            return

        if self.prev_des is None:
            self.prev_kp, self.prev_des = kp, des
            self.vo_valid = False
            return

        # ── knnMatch + Lowe's ratio test ─────────────────────
        raw_matches = self.bf.knnMatch(self.prev_des, des, k=2)

        good = []
        for pair in raw_matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.lowe_ratio * n.distance:
                good.append(m)

        if len(good) < self.min_matches:
            self.prev_kp, self.prev_des = kp, des
            self.vo_valid = False
            return

        # extract matched point coordinates
        pts_prev = np.float32([self.prev_kp[m.queryIdx].pt for m in good])
        pts_curr = np.float32([kp[m.trainIdx].pt for m in good])

        # ── Essential matrix + recoverPose ───────────────────
        E, mask_e = cv2.findEssentialMat(
            pts_prev, pts_curr, self.K,
            method=cv2.RANSAC, prob=0.999, threshold=1.0)

        if E is None:
            self.prev_kp, self.prev_des = kp, des
            self.vo_valid = False
            return

        n_inliers, R, t, mask_p = cv2.recoverPose(
            E, pts_prev, pts_curr, self.K)

        if n_inliers < self.min_matches:
            self.prev_kp, self.prev_des = kp, des
            self.vo_valid = False
            return

        # ── extract yaw from rotation matrix ─────────────────
        self.vo_delta_theta = math.atan2(R[1, 0], R[0, 0])

        # ── translation direction (unit vector) ──────────────
        # t is [3x1], only direction is valid (no scale)
        # camera z-forward → robot x-forward, camera x-right → robot y-left
        self.vo_tx = float(t[2, 0])    # camera Z = robot forward
        self.vo_ty = float(-t[0, 0])   # camera X = robot right → negate

        self.vo_valid = True

        # store for next frame
        self.prev_kp = kp
        self.prev_des = des

    # ── main odometry fusion ─────────────────────────────────

    def update_odometry(self, event):
        """Timer callback: fuse wheel encoders + ORB VO, publish Odometry.

        Wheel encoders → (delta_s, delta_theta_enc)
        ORB VO         → (delta_theta_vo, unit translation direction)

        Heading: blend encoder and VO theta.
        Position: encoder gives distance (delta_s), VO gives direction
                  correction. When VO is valid we nudge the heading used
                  for position update; the distance always comes from
                  wheel encoders (monocular camera has no scale).
        """
        current_time = rospy.Time.now()
        dt = (current_time - self.last_time).to_sec()
        if dt <= 0:
            return

        # ── wheel encoder deltas ─────────────────────────────
        delta_left = (self.left_ticks - self.last_left_ticks) / self.ticks_per_meter
        delta_right = (self.right_ticks - self.last_right_ticks) / self.ticks_per_meter

        delta_s = (delta_right + delta_left) / 2.0
        delta_theta_enc = (delta_right - delta_left) / self.wheel_base

        # ── fuse with VO ─────────────────────────────────────
        if self.vo_valid and abs(self.vo_delta_theta) < 0.5:
            # blend heading
            delta_theta = ((1.0 - self.alpha) * delta_theta_enc +
                           self.alpha * self.vo_delta_theta)

            # use VO direction to refine movement angle
            # scale VO unit translation by encoder distance
            if delta_s > 0.001:
                vo_heading = math.atan2(self.vo_ty, self.vo_tx)
                enc_heading = self.theta + delta_theta / 2.0
                move_heading = ((1.0 - self.alpha) * enc_heading +
                                self.alpha * (self.theta + vo_heading))
                self.x += delta_s * math.cos(move_heading)
                self.y += delta_s * math.sin(move_heading)
            else:
                self.x += delta_s * math.cos(self.theta + delta_theta / 2.0)
                self.y += delta_s * math.sin(self.theta + delta_theta / 2.0)

            self.vo_valid = False  # consume correction
        else:
            # pure wheel encoder fallback
            delta_theta = delta_theta_enc
            self.x += delta_s * math.cos(self.theta + delta_theta / 2.0)
            self.y += delta_s * math.sin(self.theta + delta_theta / 2.0)

        delta_theta = delta_theta
        self.theta += delta_theta

        # ── velocities ───────────────────────────────────────
        v = delta_s / dt
        w = delta_theta / dt

        # ── publish ──────────────────────────────────────────
        odom_quat = tf.transformations.quaternion_from_euler(0, 0, self.theta)

        odom_msg = Odometry()
        odom_msg.header.stamp = current_time
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = f"{self._vehicle_name}/base_link"
        odom_msg.pose.pose = Pose(
            Point(self.x, self.y, 0.0),
            Quaternion(*odom_quat))
        odom_msg.twist.twist = Twist(
            Vector3(v, 0, 0),
            Vector3(0, 0, w))

        if not rospy.is_shutdown():
            try:
                self.pub_odom.publish(odom_msg)
            except Exception:
                pass

        # ── store for next cycle ─────────────────────────────
        self.last_left_ticks = self.left_ticks
        self.last_right_ticks = self.right_ticks
        self.last_time = current_time


if __name__ == "__main__":
    node = OdometryNode("odometry_node")
    rospy.spin()