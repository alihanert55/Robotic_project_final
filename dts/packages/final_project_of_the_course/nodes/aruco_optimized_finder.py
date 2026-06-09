#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import tf.transformations
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from geometry_msgs.msg import PoseArray, Pose
from cv_bridge import CvBridge


class CustomArucoTracker:
    def __init__(self):
        self.bridge = CvBridge()

        self.marker_size = 0.065

        self.camera_matrix = None
        self.dist_coeffs = None

        s = self.marker_size / 2.0
        self.obj_points = np.array([
            [-s,  s, 0], [ s,  s, 0],
            [ s, -s, 0], [-s, -s, 0]
        ], dtype=np.float32)

        self.aruco_dict, self.parameters, self.detector, self.is_new_cv2 = self._make_detector()

        self.image_sub = rospy.Subscriber("/wolf/camera_node/image/compressed", CompressedImage, self.image_callback,
                                          queue_size=1, buff_size=2**24)
        self.info_sub  = rospy.Subscriber("/wolf/camera_node/camera_info", CameraInfo, self.info_callback)

        self.image_pub = rospy.Publisher("/aruco_tracker/result", Image, queue_size=1)
        self.poses_pub = rospy.Publisher("/aruco_tracker/poses", PoseArray, queue_size=1)

    def _make_detector(self):
        try:
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
            parameters = cv2.aruco.DetectorParameters()
            parameters.polygonalApproxAccuracyRate = 0.15
            parameters.minMarkerPerimeterRate = 0.00001
            parameters.maxMarkerPerimeterRate = 100.0
            parameters.adaptiveThreshWinSizeMin = 3
            parameters.adaptiveThreshWinSizeMax = 85
            parameters.errorCorrectionRate = 0.9
            parameters.adaptiveThreshWinSizeStep = 10
            detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
            return aruco_dict, parameters, detector, True
        except AttributeError:
            aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11)
            parameters = cv2.aruco.DetectorParameters_create()
            parameters.polygonalApproxAccuracyRate = 0.15
            parameters.minMarkerPerimeterRate = 0.00001
            parameters.maxMarkerPerimeterRate = 100.0
            parameters.adaptiveThreshWinSizeMin = 3
            parameters.adaptiveThreshWinSizeMax = 85
            parameters.errorCorrectionRate = 0.9
            parameters.adaptiveThreshWinSizeStep = 10
            return aruco_dict, parameters, None, False

    def info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.K).reshape(3, 3)
            self.dist_coeffs   = np.array(msg.D)
            self.info_sub.unregister()

    def image_callback(self, data):
        if self.camera_matrix is None:
            return

        try:
            np_arr = np.frombuffer(data.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_image is None:
                return
        except Exception:
            return

        gray    = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        out_img = cv_image.copy()

        if self.is_new_cv2:
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.parameters)

        pose_array = PoseArray()
        pose_array.header.stamp    = data.header.stamp
        pose_array.header.frame_id = data.header.frame_id or "camera_optical_frame"

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(out_img, corners, ids)
            for i in range(len(ids)):
                _, rvec, tvec = cv2.solvePnP(
                    self.obj_points, corners[i][0],
                    self.camera_matrix, self.dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE)

                pose = self._build_pose(tvec, rvec)
                pose_array.poses.append(pose)

                cv2.drawFrameAxes(
                    out_img, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.1)
                txt = f"ID:{ids[i][0]} Z:{tvec[2][0]:.2f}m"
                cv2.putText(out_img, txt,
                            (int(corners[i][0][0][0]), int(corners[i][0][0][1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        self.poses_pub.publish(pose_array)

        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(out_img, "bgr8"))
        except Exception:
            pass

    def _build_pose(self, tvec, rvec) -> Pose:
        p = Pose()
        p.position.x = tvec[0][0]
        p.position.y = tvec[1][0]
        p.position.z = tvec[2][0]

        rmat, _ = cv2.Rodrigues(rvec)
        mat = np.eye(4)
        mat[:3, :3] = rmat
        q = tf.transformations.quaternion_from_matrix(mat)
        p.orientation.x = q[0]
        p.orientation.y = q[1]
        p.orientation.z = q[2]
        p.orientation.w = q[3]
        return p


if __name__ == '__main__':
    rospy.init_node('custom_aruco_tracker')
    CustomArucoTracker()
    rospy.spin()
