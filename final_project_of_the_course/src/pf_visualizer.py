#!/usr/bin/env python3
import rospy
import tf.transformations
import cv2
import numpy as np
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Odometry
from nav_msgs.msg import Path
from sensor_msgs.msg import Image

class ParticleFilterVisualizer:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('pf_visualizer_node', anonymous=True)
        
        # Image publisher and CV bridge setup
        self.image_pub = rospy.Publisher('/vis/pf_image', Image, queue_size=10)
        self.bridge = CvBridge()
        
        # Lists to store path history and active particles
        self.odom_path = []
        self.pf_path = []
        self.particles = []
        
        # Grid visual settings
        self.img_size = 800
        self.min_xy = -2.0
        self.max_xy = 2.0
        self.scale = self.img_size / (self.max_xy - self.min_xy)
        
        # Ground-truth landmark positions (AR Tags)
        self.tag_positions = [
            (-0.274, 1.280), (0.263, 1.281),
            (0.647, -1.241), (-0.938, -1.240),
            (1.277, 0.468), (1.280, -0.746),
            (-1.238, 0.697), (-1.240, -0.719)
        ]

        # Subscribers
        rospy.Subscriber('/odom', Odometry, self.odom_callback)
        rospy.Subscriber('/pf/estimated_pose', PoseStamped, self.pf_pose_callback)
        
        # Periodic update loop running at 10Hz (0.1s duration)
        rospy.Timer(rospy.Duration(0.1), self.timer_callback)

    def world_to_pixel(self, x, y):
        # Transform real-world coordinate (meters) to pixel coordinate (image frame)
        px = int((x - self.min_xy) * self.scale)
        py = int((self.max_xy - y) * self.scale)
        return px, py

    def odom_callback(self, msg):
        # Record robot position from wheel odometry
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.odom_path.append((x, y))

    def pf_pose_callback(self, msg):
        # Record robot position estimated by the particle filter
        x = msg.pose.position.x
        y = msg.pose.position.y
        self.pf_path.append((x, y))

    def publish_particles(self, particles_list):
        self.particles = particles_list

    def draw_grid(self, img):
        # Draw reference grid lines
        for i in range(int(self.min_xy), int(self.max_xy) + 1):
            px, _ = self.world_to_pixel(i, 0)
            cv2.line(img, (px, 0), (px, self.img_size), (220, 220, 220), 1)
            _, py = self.world_to_pixel(0, i)
            cv2.line(img, (0, py), (self.img_size, py), (220, 220, 220), 1)
        
        # Draw central axes
        px, py = self.world_to_pixel(0, 0)
        cv2.line(img, (px, 0), (px, self.img_size), (150, 150, 150), 2)
        cv2.line(img, (0, py), (self.img_size, py), (150, 150, 150), 2)

    def timer_callback(self, event):
        # Initialize canvas as a blank white image
        img = np.ones((self.img_size, self.img_size, 3), dtype=np.uint8) * 255
        
        self.draw_grid(img)

        # Draw AR tags as blue squares
        for tx, ty in self.tag_positions:
            px, py = self.world_to_pixel(tx, ty)
            tag_size_px = int(0.2 * self.scale)
            half_s = tag_size_px // 2
            cv2.rectangle(img, (px - half_s, py - half_s), (px + half_s, py + half_s), (255, 0, 0), -1)

        # Draw odometry trajectory in red
        if len(self.odom_path) > 1:
            points = np.array([self.world_to_pixel(x, y) for x, y in self.odom_path], dtype=np.int32)
            cv2.polylines(img, [points], False, (0, 0, 255), 2)

        # Draw estimated PF trajectory in green
        if len(self.pf_path) > 1:
            points = np.array([self.world_to_pixel(x, y) for x, y in self.pf_path], dtype=np.int32)
            cv2.polylines(img, [points], False, (0, 255, 0), 2)

        # Draw individual particles as arrows with weight-based colors (blue to red)
        if self.particles:
            max_weight = max([p['weight'] for p in self.particles]) if self.particles else 1.0
            for p in self.particles:
                px, py = self.world_to_pixel(p['x'], p['y'])
                
                # Compute color transition from blue (lowest weight) to red (highest weight)
                norm_weight = p['weight'] / max_weight if max_weight > 0 else 0.0
                r = int(norm_weight * 255)
                g = 0
                b = int((1.0 - norm_weight) * 255)
                
                # Compute arrow end point representing the orientation
                length = int(0.15 * self.scale)
                end_x = int(px + length * np.cos(p['theta']))
                end_y = int(py - length * np.sin(p['theta']))
                
                cv2.arrowedLine(img, (px, py), (end_x, end_y), (b, g, r), 1, tipLength=0.3)
                cv2.circle(img, (px, py), 2, (b, g, r), -1)

        cv2.putText(img, "Mavi Kareler: AR Tagler", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(img, "Kirmizi Cizgi: Odom Yolu", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, "Yesil Cizgi: PF Yolu", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, "Oklar: Partikuller (Mavi->Kirmizi = Agirlik)", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Publish the final image message to ROS topic
        try:
            ros_image = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
            self.image_pub.publish(ros_image)
        except Exception as e:
            rospy.logerr("Goruntu yayinlanamadi: %s", str(e))

if __name__ == '__main__':
    try:
        visualizer = ParticleFilterVisualizer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass