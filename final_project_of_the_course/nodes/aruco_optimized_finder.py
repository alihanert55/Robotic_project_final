#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import tf.transformations
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

class CustomArucoTracker:
    def __init__(self):
        self.bridge = CvBridge()
        
        # Marker size is assumed to be 20cm based on the Gazebo environment.
        self.marker_size = 0.156  
        self.target_id = 42

        self.camera_matrix = None
        self.dist_coeffs = None

        # 3D corner coordinates of the marker are defined for solvePnP calculations.
        s = self.marker_size / 2.0
        self.obj_points = np.array([
            [-s,  s, 0], [ s,  s, 0],
            [ s, -s, 0], [-s, -s, 0]
        ], dtype=np.float32)

        # ArUco detector is initialized with custom threshold parameters.
        self.aruco_dict, self.parameters, self.detector, self.is_new_cv2 = self._make_detector()

        # ROS topics are subscribed and published.
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback)
        self.info_sub = rospy.Subscriber("/camera/rgb/camera_info", CameraInfo, self.info_callback)

        self.image_pub = rospy.Publisher("/aruco_tracker/result", Image, queue_size=10)
        self.pose_pub = rospy.Publisher("/aruco_tracker/pose", PoseStamped, queue_size=10)

    def _make_detector(self):
        # Custom parameters are applied to handle motion blur and distance.
        # Compatibility is maintained for both OpenCV < 4.7 and >= 4.7.
        try:
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
            parameters = cv2.aruco.DetectorParameters()
            parameters.polygonalApproxAccuracyRate = 0.1
            parameters.minMarkerPerimeterRate = 0.000001
            parameters.maxMarkerPerimeterRate = 100.0
            parameters.adaptiveThreshWinSizeMin = 65
            parameters.adaptiveThreshWinSizeMax = 150
            parameters.errorCorrectionRate = 0.7
            parameters.adaptiveThreshWinSizeStep = 10
            detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
            return aruco_dict, parameters, detector, True
        except AttributeError:
            aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)
            parameters = cv2.aruco.DetectorParameters_create()
            parameters.polygonalApproxAccuracyRate = 0.1
            parameters.minMarkerPerimeterRate = 0.0000001
            parameters.maxMarkerPerimeterRate = 100.0
            parameters.adaptiveThreshWinSizeMin = 65
            parameters.adaptiveThreshWinSizeMax = 150
            parameters.errorCorrectionRate = 0.7
            parameters.adaptiveThreshWinSizeStep = 10
            return aruco_dict, parameters, None, False

    def info_callback(self, msg):
        # Camera calibration matrices are saved once, and the subscription is unregistered.
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.K).reshape(3, 3)
            self.dist_coeffs = np.array(msg.D)
            self.info_sub.unregister()  

    def image_callback(self, data):
        # Image processing is skipped if camera calibration is not yet received.
        if self.camera_matrix is None: return
        
        try: 
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except: 
            return

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        out_img = cv_image.copy()

        # Markers are detected based on the available OpenCV version.
        if self.is_new_cv2: 
            corners, ids, _ = self.detector.detectMarkers(gray)
        else: 
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(out_img, corners, ids)
            for i in range(len(ids)):
                if ids[i][0] == self.target_id:
                    # 3D pose is estimated using the IPPE Square method.
                    _, rvec, tvec = cv2.solvePnP(
                        self.obj_points, corners[i][0],
                        self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)

                    # Pose message is constructed and visual axes are drawn.
                    self.publish_pose(tvec, rvec, data.header)
                    cv2.drawFrameAxes(out_img, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.1)

                    txt = f"ID:42 Z:{tvec[2][0]:.2f}m"
                    cv2.putText(out_img, txt, (int(corners[i][0][0][0]), int(corners[i][0][0][1])-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Annotated image is published for debugging purposes.
        try: 
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(out_img, "bgr8"))
        except: 
            pass

    def publish_pose(self, tvec, rvec, header):
        # PoseStamped message is populated with estimated translations.
        p = PoseStamped()
        p.header.stamp = header.stamp
        p.header.frame_id = header.frame_id or "camera_rgb_optical_frame"
        
        p.pose.position.x = tvec[0][0]
        p.pose.position.y = tvec[1][0]
        p.pose.position.z = tvec[2][0]

        # Rotation vector is converted to a 4x4 transformation matrix, then to a quaternion.
        rmat, _ = cv2.Rodrigues(rvec)
        mat = np.eye(4)
        mat[:3, :3] = rmat
        q = tf.transformations.quaternion_from_matrix(mat)
        
        p.pose.orientation.x = q[0]
        p.pose.orientation.y = q[1]
        p.pose.orientation.z = q[2]
        p.pose.orientation.w = q[3]

        self.pose_pub.publish(p)

if __name__ == '__main__':
    # Node is initialized and execution is blocked until interrupted.
    rospy.init_node('custom_aruco_tracker')
    CustomArucoTracker()
    rospy.spin()