from math import sin, cos
import numpy as np
from matplotlib.figure import Figure
from helper_fcns.utils import EndEffector, rotm_to_euler, dh_to_matrix

class TwoDOFRobot():
    """
    Represents a 2-degree-of-freedom (DOF) robot arm with two joints and one end effector.
    Includes methods for calculating forward kinematics (FPK), inverse kinematics (IPK),
    and velocity kinematics (VK).

    Attributes:
        l1 (float): Length of the first arm segment.
        l2 (float): Length of the second arm segment.
        theta (list): Joint angles.
        theta_limits (list): Joint limits for each joint.
        ee (EndEffector): The end effector object.
        points (list): List of points representing the robot's configuration.
        num_dof (int): The number of degrees of freedom (2 for this robot).
    """

    def __init__(self):
        """
        Initializes a 2-DOF robot with default arm segment lengths and joint angles.
        """
        self.l1 = 0.30  # Length of the first arm segment
        self.l2 = 0.25  # Length of the second arm segment

        self.theta = [0.0, 0.0]  # Joint angles (in radians)
        self.theta_limits = [[-np.pi, np.pi], [-np.pi + 0.261, np.pi - 0.261]]  # Joint limits

        self.ee = EndEffector()  # The end-effector object
        self.num_dof = 2  # Number of degrees of freedom
        self.points = [None] * (self.num_dof + 1)  # List to store robot points
        
        self.T = np.zeros((self.num_dof, 4, 4))  # Transformation matrices


    def calc_forward_kinematics(self, theta: list, radians=False):
        """
        Calculates the forward kinematics for the robot based on the joint angles.

        Args:
            theta (list): Joint angles.
            radians (bool, optional): Whether the angles are in radians or degrees. Defaults to False.
        """
        if not radians:
            theta = np.radians(theta)  # Convert to radians if the input is in degrees

        # Update the robot configuration (i.e., the positions of the joints and end effector)
        self.calc_robot_points()


    def calc_inverse_kinematics(self, EE: EndEffector, soln=0):
        """
        Calculates the inverse kinematics (IK) for a given end effector position.

        Args:
            EE (EndEffector): The end effector object containing the target position (x, y).
            soln (int, optional): The solution branch to use. Defaults to 0 (first solution).
        """
        elbow_pos = 1
        if(soln == 1):
            elbow_pos = -elbow_pos

        x, y = EE.x, EE.y
        l1, l2 = self.l1, self.l2

        L = np.sqrt(x ** 2 + y ** 2)
        
        beta = np.arccos((l1 ** 2 + l2 ** 2 - L ** 2) / (2 * l1 * l2))
        self.theta[1] = np.pi + (elbow_pos * beta)
        print(elbow_pos)

        alpha = np.arctan2(l2 * sin(self.theta[1]), l1 + l2 * cos(self.theta[1]))
        gamma = np.arctan2(y, x)

        self.theta[0] = gamma - alpha

        print(self.theta)
        
        # Calculate robot points based on the updated joint angles
        self.calc_robot_points()


    def calc_numerical_ik(self, EE: EndEffector, tol=0.01, ilimit=50):
        """
        Calculates numerical inverse kinematics (IK) based on input end effector coordinates.

        Args:
            EE (EndEffector): The end effector object containing the target position (x, y).
            tol (float, optional): The tolerance for the solution. Defaults to 0.01.
            ilimit (int, optional): The maximum number of iterations. Defaults to 50.
        """
        
        x, y = EE.x, EE.y
        
        ########################################

        # insert your code here

        ########################################

        self.calc_robot_points()


    def calc_velocity_kinematics(self, vel: list):
        """
        Calculates the velocity kinematics for the robot based on the given velocity input.

        Args:
            vel (list): The velocity vector for the end effector [vx, vy].
        """
        
        ########################################

        # insert your code here

        ########################################

        # Update robot points based on the new joint angles
        self.calc_robot_points()


    def calc_robot_points(self):
        """
        Calculates the positions of the robot's joints and the end effector.

        Updates the `points` list, storing the coordinates of the base, shoulder, elbow, and end effector.
        """

        self.T[0] = dh_to_matrix([self.theta[0], 0, self.l1, 0])
        self.T[1] = dh_to_matrix([self.theta[1], 0, self.l2, 0])

        # Base position
        self.points[0] = [0.0, 0.0, 0.0, 1.0]
        # Shoulder joint
        self.points[1] = self.T[0] @ self.points[0]  # Shoulder position
        # Elbow joint
        self.points[2] = self.T[0] @ self.T[1] @ self.points[0]  # End-effector position

        self.points[0] = self.points[0][:3]
        self.points[1] = self.points[1][:3]
        self.points[2] = self.points[2][:3]

        # Update end effector position
        self.ee.x = self.points[2][0]
        self.ee.y = self.points[2][1]
        self.ee.z = self.points[2][2]
        self.ee.rotz = self.theta[0] + self.theta[1]

        # End effector axes
        self.EE_axes = np.zeros((3, 3))
        self.EE_axes[0] = np.array([cos(self.theta[0] + self.theta[1]), sin(self.theta[0] + self.theta[1]), 0]) * 0.075 + self.points[2]
        self.EE_axes[1] = np.array([-sin(self.theta[0] + self.theta[1]), cos(self.theta[0] + self.theta[1]), 0]) * 0.075 + self.points[2]
        self.EE_axes[2] = np.array([0, 0, 1]) * 0.075 + self.points[2]
