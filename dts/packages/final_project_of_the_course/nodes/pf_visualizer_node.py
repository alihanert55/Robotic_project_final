#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseArray
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray


class ParticleFilterVisualizer:
    def __init__(self):
        rospy.init_node('pf_visualizer_node', anonymous=True)

        self.bridge    = CvBridge()
        self.image_pub = rospy.Publisher('/vis/pf_image', Image, queue_size=10)

        self.odom_path = []
        self.pf_path   = []
        self.particles = []   # list of (x, y, theta)
        self.weights   = []   # parallel list of floats
        self.neff      = 0.0
        self.pf_est    = (0.0, 0.0, 0.0)   # (x, y, theta) — latest PF estimate

        self.img_size = 800
        self.min_xy   = -0.80
        self.max_xy   =  0.80
        self.scale    = self.img_size / (self.max_xy - self.min_xy)

        # Room bounds — 1.24m x 0.80m, origin at center
        self.room_x_min = -0.62
        self.room_x_max =  0.62
        self.room_y_min = -0.40
        self.room_y_max =  0.40

        self.tag_positions = [
            (-0.22,  0.40), ( 0.22,  0.40),  # north
            (-0.62,  0.15), (-0.62, -0.15),  # west
            ( 0.62,  0.15), ( 0.62, -0.15),  # east
            (-0.22, -0.40), ( 0.22, -0.40),  # south
        ]

        rospy.Subscriber('/odometry',         Odometry,         self._odom_cb,      queue_size=1)
        rospy.Subscriber('/pf_path',          Path,             self._pf_path_cb,   queue_size=1)
        rospy.Subscriber('/particles',        PoseArray,        self._particles_cb, queue_size=1)
        rospy.Subscriber('/particle_weights', Float32MultiArray, self._weights_cb,  queue_size=1)
        rospy.Subscriber('/pf_neff',          Float32MultiArray, self._neff_cb,     queue_size=1)

        rospy.Timer(rospy.Duration(0.1), self._timer_cb)

    # ------------------------------------------------------------------
    def _odom_cb(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.odom_path.append((x, y))

    def _pf_path_cb(self, msg):
        if msg.poses:
            p = msg.poses[-1].pose.position
            self.pf_path.append((p.x, p.y))

    def _particles_cb(self, msg: PoseArray):
        pts = []
        for pose in msg.poses:
            q = pose.orientation
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            pts.append((pose.position.x, pose.position.y, np.arctan2(siny, cosy)))
        self.particles = pts

    def _weights_cb(self, msg: Float32MultiArray):
        self.weights = list(msg.data)

    def _neff_cb(self, msg: Float32MultiArray):
        if len(msg.data) >= 4:
            self.neff   = msg.data[0]
            self.pf_est = (msg.data[1], msg.data[2], msg.data[3])

    # ------------------------------------------------------------------
    def world_to_pixel(self, x, y):
        px = int((x - self.min_xy) * self.scale)
        py = int((self.max_xy - y) * self.scale)
        return px, py

    def _draw_grid(self, img):
        for i in range(int(self.min_xy), int(self.max_xy) + 1):
            px, _  = self.world_to_pixel(i, 0)
            _,  py = self.world_to_pixel(0, i)
            cv2.line(img, (px, 0),           (px, self.img_size), (220, 220, 220), 1)
            cv2.line(img, (0,  py),          (self.img_size, py), (220, 220, 220), 1)
        ox, oy = self.world_to_pixel(0, 0)
        cv2.line(img, (ox, 0), (ox, self.img_size), (180, 180, 180), 2)
        cv2.line(img, (0, oy), (self.img_size, oy), (180, 180, 180), 2)

    def _draw_room_boundary(self, img):
        tl = self.world_to_pixel(self.room_x_min, self.room_y_max)
        br = self.world_to_pixel(self.room_x_max, self.room_y_min)
        cv2.rectangle(img, tl, br, (80, 80, 80), 2)

    def _draw_tags(self, img):
        half = int(0.09 * self.scale)
        for tx, ty in self.tag_positions:
            px, py = self.world_to_pixel(tx, ty)
            cv2.rectangle(img, (px - half, py - half), (px + half, py + half), (200, 80, 0), -1)
            cv2.rectangle(img, (px - half, py - half), (px + half, py + half), (255, 140, 0), 1)

    def _draw_particles(self, img):
        if not self.particles:
            return
        n = len(self.particles)
        weights = self.weights if len(self.weights) == n else [1.0 / n] * n
        w_arr = np.array(weights, dtype=float)
        max_w = w_arr.max()
        if max_w <= 0:
            max_w = 1.0
        w_arr /= max_w

        arrow_len = int(0.10 * self.scale)
        for i, (x, y, theta) in enumerate(self.particles):
            px, py = self.world_to_pixel(x, y)
            nw = float(w_arr[i])
            # blue (low weight) → red (high weight)
            color = (int((1 - nw) * 200), 30, int(nw * 220))
            ex = int(px + arrow_len * np.cos(theta))
            ey = int(py - arrow_len * np.sin(theta))
            cv2.arrowedLine(img, (px, py), (ex, ey), color, 1, tipLength=0.4)
            cv2.circle(img, (px, py), 2, color, -1)

    def _draw_pf_estimate(self, img):
        x, y, theta = self.pf_est
        px, py = self.world_to_pixel(x, y)
        # Filled circle
        cv2.circle(img, (px, py), 12, (0, 200, 0), -1)
        cv2.circle(img, (px, py), 12, (0, 100, 0), 2)
        # Direction arrow
        arrow_len = int(0.18 * self.scale)
        ex = int(px + arrow_len * np.cos(theta))
        ey = int(py - arrow_len * np.sin(theta))
        cv2.arrowedLine(img, (px, py), (ex, ey), (0, 80, 0), 3, tipLength=0.3)

    def _draw_overlay(self, img):
        n = rospy.get_param('/particle_filter_node/n_particles', len(self.particles))
        x, y, theta = self.pf_est
        lines = [
            f"PF Estimate: x={x:.2f}  y={y:.2f}  th={np.degrees(theta):.1f} deg",
            f"N_eff: {self.neff:.0f} / {n}",
            f"Particles: {len(self.particles)}",
        ]
        pad = 10
        box_h = len(lines) * 28 + pad * 2
        box_w = 420
        cv2.rectangle(img, (pad, pad), (pad + box_w, pad + box_h), (240, 240, 240), -1)
        cv2.rectangle(img, (pad, pad), (pad + box_w, pad + box_h), (100, 100, 100), 1)
        for i, line in enumerate(lines):
            cv2.putText(img, line, (pad + 8, pad + 22 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1, cv2.LINE_AA)

    def _draw_legend(self, img):
        items = [
            ("AR Tag",        (200, 80,   0)),
            ("Odometry path", (  0,  0, 220)),
            ("PF path",       (  0, 180,  0)),
            ("PF estimate",   (  0, 200,  0)),
            ("Particles",     ( 80, 30,  180)),
        ]
        bx = self.img_size - 200
        by = 10
        cv2.rectangle(img, (bx - 5, by), (self.img_size - 5, by + len(items) * 26 + 10),
                      (240, 240, 240), -1)
        cv2.rectangle(img, (bx - 5, by), (self.img_size - 5, by + len(items) * 26 + 10),
                      (100, 100, 100), 1)
        for i, (label, color) in enumerate(items):
            y = by + 22 + i * 26
            cv2.rectangle(img, (bx, y - 10), (bx + 18, y + 4), color, -1)
            cv2.putText(img, label, (bx + 24, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (30, 30, 30), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    def _timer_cb(self, _event):
        img = np.ones((self.img_size, self.img_size, 3), dtype=np.uint8) * 255
        self._draw_grid(img)
        self._draw_room_boundary(img)
        self._draw_tags(img)

        # Odometry path — red
        if len(self.odom_path) > 1:
            pts = np.array([self.world_to_pixel(x, y)
                            for x, y in self.odom_path], dtype=np.int32)
            cv2.polylines(img, [pts], False, (0, 0, 220), 2)

        # PF path — green
        if len(self.pf_path) > 1:
            pts = np.array([self.world_to_pixel(x, y)
                            for x, y in self.pf_path], dtype=np.int32)
            cv2.polylines(img, [pts], False, (0, 180, 0), 2)

        self._draw_particles(img)
        self._draw_pf_estimate(img)
        self._draw_overlay(img)
        self._draw_legend(img)

        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(img, encoding='bgr8'))
        except Exception as e:
            rospy.logerr("Goruntu yayinlanamadi: %s", e)


if __name__ == '__main__':
    try:
        ParticleFilterVisualizer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass