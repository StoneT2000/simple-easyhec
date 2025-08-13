import numpy as np


def ros2opencv(transformation: np.ndarray):
    """
    Convert a ROS convention to opencv convention transformation.
    """
    ros2opencv = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]).T
    opencv_transformation = ros2opencv @ np.linalg.inv(transformation)
    return opencv_transformation


def opencv2ros(transformation: np.ndarray):
    """
    Convert an opencv convention transformation to a ROS convention transformation.
    """
    ros2opencv = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]).T
    ros_transformation = np.linalg.inv(ros2opencv.T @ transformation)
    return ros_transformation
