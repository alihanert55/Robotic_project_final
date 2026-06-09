#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import math
import tf.transformations

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, PoseArray, Pose
from std_msgs.msg import Float32MultiArray

# --- TUNABLE PARAMETERS ---
NUM_PARTICLES  = 500
SIGMA_DIST     = 0.5
SIGMA_BEARING  = 0.5
CAMERA_X_OFFSET = 0.076

# Odometry noise
NOISE_X     = 0.05
NOISE_Y     = 0.05
NOISE_THETA = 0.05

# Innovation gating thresholds (base values, scaled by spatial spread)
INNOVATION_DIST_THRESH = 0.03   # meters
INNOVATION_BEAR_THRESH = 0.05   # radians
MIN_UPDATE_INTERVAL    = 0.05   # seconds (~20 Hz ceiling)
# --------------------------

# Sensor model normalization constant.
_SENSOR_NORM = 1.0 / (2.0 * math.pi * SIGMA_DIST * SIGMA_BEARING)


class ParticleFilterNode:
    def __init__(self):
        rospy.init_node('particle_filter_node')

        self.num_particles = rospy.get_param('~num_particles', NUM_PARTICLES)

        tag_config = rospy.get_param('/tags', [])
        if not tag_config:
            rospy.logwarn("AR tags could not be read from '/tags'. Make sure tag_map.yaml is loaded.")
        self.tags = [{"x": t["x"], "y": t["y"]} for t in tag_config]

        bounds = rospy.get_param('~room_bounds', [-1.5, -1.5, 1.5, 1.5])
        self.room_bounds = bounds  # [x_min, y_min, x_max, y_max]

        self.particles = self._initialize_particles(self.num_particles)
        self.weights   = np.ones(self.num_particles) / self.num_particles

        self.last_odom               = None
        self.last_aruco_update       = 0.0
        self.last_accepted_measurement = None  # (distance, bearing) of last accepted obs

        self.pf_path_msg = Path()
        self.pf_path_msg.header.frame_id = "odom"

        # Publishers
        self.pub_particles = rospy.Publisher('/particles',        PoseArray,        queue_size=10)
        self.pub_weights   = rospy.Publisher('/particle_weights', Float32MultiArray, queue_size=10)
        self.pub_neff      = rospy.Publisher('/pf_neff',          Float32MultiArray, queue_size=10)
        self.pub_spread    = rospy.Publisher('/pf_spread',        Float32MultiArray, queue_size=10)
        self.pub_path      = rospy.Publisher('/pf_path',          Path,             queue_size=10)

        # Subscribers
        rospy.Subscriber('/odom',                Odometry, self.odom_callback)

        rospy.Subscriber('/aruco_tracker/poses', PoseArray, self.aruco_callback)

        rospy.Timer(rospy.Duration(0.1), self.publish_state)

        rospy.loginfo("Particle Filter with Innovation Gating started. "
                      "Room bounds: %s", self.room_bounds)

    # =========================================================================
    # INITIALIZATION
    # =========================================================================
    def _initialize_particles(self, num):
        x_min, y_min, x_max, y_max = self.room_bounds
        particles = np.empty((num, 3))
        particles[:, 0] = np.random.uniform(x_min, x_max, num)
        particles[:, 1] = np.random.uniform(y_min, y_max, num)
        particles[:, 2] = np.random.uniform(-math.pi, math.pi, num)
        return particles

    @staticmethod
    def _wrap_angle(angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    # =========================================================================
    # MOTION MODEL
    # =========================================================================
    def odom_callback(self, msg):
        pose = msg.pose.pose
        q    = [pose.orientation.x, pose.orientation.y,
                pose.orientation.z, pose.orientation.w]
        _, _, yaw = tf.transformations.euler_from_quaternion(q)

        current_odom = np.array([pose.position.x, pose.position.y, yaw])

        if self.last_odom is not None:
            dx     = current_odom[0] - self.last_odom[0]
            dy     = current_odom[1] - self.last_odom[1]
            dtheta = self._wrap_angle(current_odom[2] - self.last_odom[2])

            trans    = math.sqrt(dx**2 + dy**2)
            move_dir = math.atan2(dy, dx) - self.last_odom[2]

            local_dx = trans * math.cos(move_dir)
            local_dy = trans * math.sin(move_dir)

            if trans > 0.001 or abs(dtheta) > 0.001:
                self._predict(local_dx, local_dy, dtheta, trans)

        self.last_odom = current_odom

    def _predict(self, local_dx, local_dy, dtheta, trans):
        std_x     = max(0.001, NOISE_X     * trans)
        std_y     = max(0.001, NOISE_Y     * trans)
        std_theta = max(0.001, NOISE_THETA * abs(dtheta))

        noise_x     = np.random.normal(0, std_x,     self.num_particles)
        noise_y     = np.random.normal(0, std_y,     self.num_particles)
        noise_theta = np.random.normal(0, std_theta, self.num_particles)

        cos_theta = np.cos(self.particles[:, 2])
        sin_theta = np.sin(self.particles[:, 2])

        self.particles[:, 0] += (local_dx + noise_x) * cos_theta \
                               - (local_dy + noise_y) * sin_theta
        self.particles[:, 1] += (local_dx + noise_x) * sin_theta \
                               + (local_dy + noise_y) * cos_theta
        self.particles[:, 2] += dtheta + noise_theta
        self.particles[:, 2]  = self._wrap_angle(self.particles[:, 2])

    # =========================================================================
    # SENSOR MODEL — Innovation Gating + Spread Adaptive
    # =========================================================================
    def aruco_callback(self, msg: PoseArray):
        """Handle one PoseArray message — may contain 0..N detections from a
        single camera frame.  Each Pose is in the camera optical frame."""
        if not self.tags or not msg.poses:
            return

        current_time = rospy.Time.now().to_sec()

        # LAYER 1: Hard rate limit (~20 Hz ceiling)
        if (current_time - self.last_aruco_update) < MIN_UPDATE_INTERVAL:
            return

        # Convert every detection from camera optical frame to base_link (range, bearing).
        observations = []
        for pose in msg.poses:
            z_opt = pose.position.z   # forward in optical frame
            x_opt = pose.position.x   # right   in optical frame
            x_base = z_opt + CAMERA_X_OFFSET
            y_base = -x_opt
            dist    = math.sqrt(x_base**2 + y_base**2)
            bearing = math.atan2(y_base, x_base)
            observations.append((dist, bearing))

        # Innovation gate on the FIRST (or only) observation for rate control.
        # If all detections look stale, skip; otherwise process the full list.
        first_dist, first_bear = observations[0]
        if self.last_accepted_measurement is not None:
            d_dist = abs(first_dist - self.last_accepted_measurement[0])
            d_bear = abs(self._wrap_angle(
                first_bear - self.last_accepted_measurement[1]))

            # LAYER 3: Spread-adaptive threshold scaling
            _, _, spread = self._get_spatial_spread()
            scale = np.clip(1.0 - 1.56 * (spread - 0.05), 0.3, 1.0)

            if (d_dist < INNOVATION_DIST_THRESH * scale and
                    d_bear < INNOVATION_BEAR_THRESH * scale):
                return

        # Measurement accepted — run update with all observations from this frame.
        self.last_aruco_update         = current_time
        self.last_accepted_measurement = (first_dist, first_bear)

        self._update(observations)

        n_eff = 1.0 / (np.sum(self.weights**2) + 1e-10)
        if n_eff < self.num_particles * 0.5:
            self._resample()

    # =========================================================================
    # WEIGHT UPDATE — Multi-hypothesis likelihood
    # =========================================================================
    def _update(self, observations):
        """observations: list of (distance, bearing) tuples in base_link frame.

        For each observation and each particle, sum the likelihood over all
        map tags (uniform prior over which tag was seen).  Then multiply
        across independent observations.

            w_i *= prod_k  sum_j  p(z_k | x_i, tag_j)
        """
        for measured_distance, measured_bearing in observations:
            likelihoods = np.zeros(self.num_particles)

            for tag in self.tags:
                dx = tag["x"] - self.particles[:, 0]
                dy = tag["y"] - self.particles[:, 1]

                expected_distance = np.sqrt(dx**2 + dy**2)
                expected_bearing  = self._wrap_angle(
                    np.arctan2(dy, dx) - self.particles[:, 2])

                dist_error    = measured_distance - expected_distance
                bearing_error = self._wrap_angle(measured_bearing - expected_bearing)

                l = _SENSOR_NORM * \
                    np.exp(-(dist_error**2)    / (2 * SIGMA_DIST**2)) * \
                    np.exp(-(bearing_error**2) / (2 * SIGMA_BEARING**2))

                likelihoods += l

            self.weights *= likelihoods

        weight_sum = np.sum(self.weights)
        if weight_sum < 1e-10:
            self.weights = np.ones(self.num_particles) / self.num_particles
        else:
            self.weights /= weight_sum

    # =========================================================================
    # RESAMPLING — low-variance + roughening
    # =========================================================================
    def _resample(self):
        positions = (np.arange(self.num_particles) + np.random.uniform()) / self.num_particles
        cumsum    = np.cumsum(self.weights)
        indices   = np.searchsorted(cumsum, positions)
        self.particles = self.particles[indices].copy()

        # Roughening: break exact clones apart so each particle explores a
        # slightly different neighborhood after resampling.
        self.particles[:, 0] += np.random.normal(0, 0.02, self.num_particles)
        self.particles[:, 1] += np.random.normal(0, 0.02, self.num_particles)
        self.particles[:, 2] += np.random.normal(0, 0.01, self.num_particles)
        self.particles[:, 2]  = self._wrap_angle(self.particles[:, 2])

        self.weights = np.ones(self.num_particles) / self.num_particles

    # =========================================================================
    # ESTIMATION
    # =========================================================================
    def _get_estimate(self):
        w = self.weights
        x = float(np.sum(self.particles[:, 0] * w))
        y = float(np.sum(self.particles[:, 1] * w))
        cos_t = float(np.sum(np.cos(self.particles[:, 2]) * w))
        sin_t = float(np.sum(np.sin(self.particles[:, 2]) * w))
        theta = math.atan2(sin_t, cos_t)
        return x, y, theta

    def _get_spatial_spread(self):
        mean_x   = np.average(self.particles[:, 0], weights=self.weights)
        mean_y   = np.average(self.particles[:, 1], weights=self.weights)
        spread_x = np.sqrt(np.average(
            (self.particles[:, 0] - mean_x)**2, weights=self.weights))
        spread_y = np.sqrt(np.average(
            (self.particles[:, 1] - mean_y)**2, weights=self.weights))
        return spread_x, spread_y, math.sqrt(spread_x**2 + spread_y**2)

    # =========================================================================
    # STATE PUBLISHING
    # =========================================================================
    def publish_state(self, event):
        # 1. Particle cloud (PoseArray — consumed by pf_visualizer_node)
        pa = PoseArray()
        pa.header.stamp    = rospy.Time.now()
        pa.header.frame_id = "odom"
        for i in range(self.num_particles):
            p = Pose()
            p.position.x = self.particles[i, 0]
            p.position.y = self.particles[i, 1]
            q = tf.transformations.quaternion_from_euler(0, 0, self.particles[i, 2])
            p.orientation.x = q[0]
            p.orientation.y = q[1]
            p.orientation.z = q[2]
            p.orientation.w = q[3]
            pa.poses.append(p)
        self.pub_particles.publish(pa)

        # 2. Per-particle weights
        wa = Float32MultiArray()
        wa.data = self.weights.tolist()
        self.pub_weights.publish(wa)

        # 3. N_eff + estimated pose (consumed by pf_visualizer_node as /pf_neff)
        n_eff = 1.0 / (np.sum(self.weights**2) + 1e-10)
        ex, ey, etheta = self._get_estimate()
        neff_msg      = Float32MultiArray()
        neff_msg.data = [n_eff, ex, ey, etheta]
        self.pub_neff.publish(neff_msg)

        # 4. Spatial spread (real convergence metric)
        _, _, spread  = self._get_spatial_spread()
        spread_msg    = Float32MultiArray()
        spread_msg.data = [spread]
        self.pub_spread.publish(spread_msg)

        # 5. PF estimate trajectory (consumed by pf_visualizer_node as /pf_path)
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp    = rospy.Time.now()
        pose_stamped.header.frame_id = "odom"
        pose_stamped.pose.position.x = ex
        pose_stamped.pose.position.y = ey
        q_est = tf.transformations.quaternion_from_euler(0, 0, etheta)
        pose_stamped.pose.orientation.x = q_est[0]
        pose_stamped.pose.orientation.y = q_est[1]
        pose_stamped.pose.orientation.z = q_est[2]
        pose_stamped.pose.orientation.w = q_est[3]

        self.pf_path_msg.poses.append(pose_stamped)
        if len(self.pf_path_msg.poses) > 1000:
            self.pf_path_msg.poses.pop(0)
        self.pub_path.publish(self.pf_path_msg)


if __name__ == '__main__':
    try:
        ParticleFilterNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass