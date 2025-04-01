from math import sin, cos
import numpy as np
from matplotlib.figure import Figure
from helper_fcns.utils import EndEffector, rotm_to_euler, dh_to_matrix, euler_to_rotm, wraptopi

PI = 3.1415926535897932384
np.set_printoptions(precision=3)

class Robot:
    """
    Represents a robot manipulator with various kinematic configurations.
    Provides methods to calculate forward kinematics, inverse kinematics, and velocity kinematics.
    Also includes methods to visualize the robot's motion and state in 3D.

    Attributes:
        num_joints (int): Number of joints in the robot.
        ee_coordinates (list): List of end-effector coordinates.
        robot (object): The robot object (e.g., TwoDOFRobot, ScaraRobot, etc.).
        origin (list): Origin of the coordinate system.
        axes_length (float): Length of the axes for visualization.
        point_x, point_y, point_z (list): Lists to store coordinates of points for visualization.
        show_animation (bool): Whether to show the animation or not.
        plot_limits (list): Limits for the plot view.
        fig (matplotlib.figure.Figure): Matplotlib figure for 3D visualization.
        sub1 (matplotlib.axes._subplots.Axes3DSubplot): Matplotlib 3D subplot.
    """

    def __init__(self, type='2-dof', show_animation: bool=True):
        """
        Initializes a robot with a specific configuration based on the type.

        Args:
            type (str, optional): Type of robot (e.g., '2-dof', 'scara', '5-dof'). Defaults to '2-dof'.
            show_animation (bool, optional): Whether to show animation of robot movement. Defaults to True.
        """
        if type == '2-dof':
            self.num_joints = 2
            self.ee_coordinates = ['X', 'Y']
            self.robot = TwoDOFRobot()
        
        elif type == 'scara':
            self.num_joints = 3
            self.ee_coordinates = ['X', 'Y', 'Z', 'Theta']
            self.robot = ScaraRobot()

        elif type == '5-dof':
            self.num_joints = 5
            self.ee_coordinates = ['X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ']
            self.robot = FiveDOFRobot()
        
        self.origin = [0., 0., 0.]
        self.axes_length = 0.04
        self.point_x, self.point_y, self.point_z = [], [], []
        self.waypoint_x, self.waypoint_y, self.waypoint_z = [], [], []
        self.waypoint_rotx, self.waypoint_roty, self.waypoint_rotz = [], [], []
        self.show_animation = show_animation
        self.plot_limits = [0.65, 0.65, 0.8]

        if self.show_animation:
            self.fig = Figure(figsize=(12, 10), dpi=100)
            self.sub1 = self.fig.add_subplot(1,1,1, projection='3d') 
            self.fig.suptitle("Manipulator Kinematics Visualization", fontsize=16)

        # initialize figure plot
        self.init_plot()

    
    def init_plot(self):
        """Initializes the plot by calculating the robot's points and calling the plot function."""
        self.robot.calc_robot_points()
        self.plot_3D()

    
    def update_plot(self, pose=None, angles=None, soln=0, numerical=False):
        """
        Updates the robot's state based on new pose or joint angles and updates the visualization.

        Args:
            pose (EndEffector, optional): Desired end-effector pose for inverse kinematics.
            angles (list, optional): Joint angles for forward kinematics.
            soln (int, optional): The inverse kinematics solution to use (0 or 1).
            numerical (bool, optional): Whether to use numerical inverse kinematics.
        """
        if pose is not None: # Inverse kinematics case
            if not numerical:
                self.robot.calc_inverse_kinematics(pose, soln=soln)
            else:
                self.robot.calc_numerical_ik(pose, tol=0.02, ilimit=50)
        elif angles is not None: # Forward kinematics case
            self.robot.calc_forward_kinematics(angles, radians=False)
        else:
            return
        self.plot_3D()


    def move_velocity(self, vel):
        """
        Moves the robot based on a given velocity input.

        Args:
            vel (list): Velocity input for the robot.
        """
        self.robot.calc_velocity_kinematics(vel)
        self.plot_3D()
        

    def draw_line_3D(self, p1, p2, format_type: str = "k-"):
        """
        Draws a 3D line between two points.

        Args:
            p1 (list): Coordinates of the first point.
            p2 (list): Coordinates of the second point.
            format_type (str, optional): The format of the line. Defaults to "k-".
        """
        self.sub1.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], format_type)


    def draw_ref_line(self, point, axes=None, ref='xyz'):
        """
        Draws reference lines from a given point along specified axes.

        Args:
            point (list): The coordinates of the point to draw from.
            axes (matplotlib.axes, optional): The axes on which to draw the reference lines.
            ref (str, optional): Which reference axes to draw ('xyz', 'xy', or 'xz'). Defaults to 'xyz'.
        """
        line_width = 0.7
        if ref == 'xyz':
            axes.plot([point[0], self.plot_limits[0]],
                      [point[1], point[1]],
                      [point[2], point[2]], 'b--', linewidth=line_width)    # X line
            axes.plot([point[0], point[0]],
                      [point[1], self.plot_limits[1]],
                      [point[2], point[2]], 'b--', linewidth=line_width)    # Y line
            axes.plot([point[0], point[0]],
                      [point[1], point[1]],
                      [point[2], 0.0], 'b--', linewidth=line_width)         # Z line
        elif ref == 'xy':
            axes.plot([point[0], self.plot_limits[0]],
                      [point[1], point[1]], 'b--', linewidth=line_width)    # X line
            axes.plot([point[0], point[0]],
                      [point[1], self.plot_limits[1]], 'b--', linewidth=line_width)    # Y line
        elif ref == 'xz':
            axes.plot([point[0], self.plot_limits[0]],
                      [point[2], point[2]], 'b--', linewidth=line_width)    # X line
            axes.plot([point[0], point[0]],
                      [point[2], 0.0], 'b--', linewidth=line_width)         # Z line


    def plot_waypoints(self):
        """
        Plots the waypoints in the 3D visualization
        """
        # draw the points
        self.sub1.plot(self.waypoint_x, self.waypoint_y, self.waypoint_z, 'or', markersize=8)


    def update_waypoints(self, waypoints: list):
        """
        Updates the waypoints into a member variable
        """
        for i in range(len(waypoints)):
            self.waypoint_x.append(waypoints[i][0])
            self.waypoint_y.append(waypoints[i][1])
            self.waypoint_z.append(waypoints[i][2])
            # self.waypoint_rotx.append(waypoints[i][3])
            # self.waypoint_roty.append(waypoints[i][4])
            # self.waypoint_rotz.append(waypoints[i][5])


    def plot_3D(self):
        """
        Plots the 3D visualization of the robot, including the robot's links, end-effector, and reference frames.
        """        
        self.sub1.cla()
        self.point_x.clear()
        self.point_y.clear()
        self.point_z.clear()

        EE = self.robot.ee

        # draw lines to connect the points
        for i in range(len(self.robot.points)-1):
            self.draw_line_3D(self.robot.points[i], self.robot.points[i+1])

        # draw the points
        for i in range(len(self.robot.points)):
            self.point_x.append(self.robot.points[i][0])
            self.point_y.append(self.robot.points[i][1])
            self.point_z.append(self.robot.points[i][2])
        self.sub1.plot(self.point_x, self.point_y, self.point_z, marker='o', markerfacecolor='m', markersize=12)


        # draw the waypoints
        self.plot_waypoints()

        # draw the EE
        self.sub1.plot(EE.x, EE.y, EE.z, 'bo')
        # draw the base reference frame
        self.draw_line_3D(self.origin, [self.origin[0] + self.axes_length, self.origin[1], self.origin[2]], format_type='r-')
        self.draw_line_3D(self.origin, [self.origin[0], self.origin[1] + self.axes_length, self.origin[2]], format_type='g-')
        self.draw_line_3D(self.origin, [self.origin[0], self.origin[1], self.origin[2] + self.axes_length], format_type='b-')
        # draw the EE reference frame
        self.draw_line_3D([EE.x, EE.y, EE.z], self.robot.EE_axes[0], format_type='r-')
        self.draw_line_3D([EE.x, EE.y, EE.z], self.robot.EE_axes[1], format_type='g-')
        self.draw_line_3D([EE.x, EE.y, EE.z], self.robot.EE_axes[2], format_type='b-')
        # draw reference / trace lines
        self.draw_ref_line([EE.x, EE.y, EE.z], self.sub1, ref='xyz')

        # add text at bottom of window
        pose_text = "End-effector Pose:      [ "
        pose_text += f"X: {round(EE.x,4)},  "
        pose_text += f"Y: {round(EE.y,4)},  "
        pose_text += f"Z: {round(EE.z,4)},  "
        pose_text += f"RotX: {round(EE.rotx,4)},  "
        pose_text += f"RotY: {round(EE.roty,4)},  "
        pose_text += f"RotZ: {round(EE.rotz,4)}  "
        pose_text += " ]"

        theta_text = "Joint Positions (deg/m):     ["
        for i in range(self.num_joints):
            theta_text += f" {round(np.rad2deg(self.robot.theta[i]),2)}, "
        theta_text += " ]"
        
        textstr = pose_text + "\n" + theta_text
        self.sub1.text2D(0.2, 0.02, textstr, fontsize=13, transform=self.fig.transFigure)

        self.sub1.set_xlim(-self.plot_limits[0], self.plot_limits[0])
        self.sub1.set_ylim(-self.plot_limits[1], self.plot_limits[1])
        self.sub1.set_zlim(0, self.plot_limits[2])
        self.sub1.set_xlabel('x [m]')
        self.sub1.set_ylabel('y [m]')

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
        self.theta_limits = [[-PI, PI], [-PI + 0.261, PI - 0.261]]  # Joint limits

        self.ee = EndEffector()  # The end-effector object
        self.num_dof = 2  # Number of degrees of freedom
        self.points = [None] * (self.num_dof + 1)  # List to store robot points
        
        self.T = np.zeros((self.num_dof, 4, 4))  # Transformation matrices

        self.J = np.zeros((2, 3))


    def calc_forward_kinematics(self, theta: list, radians=False):
        """
        Calculates the forward kinematics for the robot based on the joint angles.

        Args:
            theta (list): Joint angles.
            radians (bool, optional): Whether the angles are in radians or degrees. Defaults to False.
        """
        if not radians:
            theta = np.radians(theta)  # Convert to radians if the input is in degrees

        self.theta = theta

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

        alpha = np.arctan2(l2 * sin(self.theta[1]), l1 + l2 * cos(self.theta[1]))
        gamma = np.arctan2(y, x)

        self.theta[0] = gamma - alpha
        
        # Calculate robot points based on the updated joint angles
        self.calc_robot_points()

    def calculate_jacobian(self):
        self.T[0] = dh_to_matrix([self.theta[0], 0, self.l1, 0])
        self.T[1] = dh_to_matrix([self.theta[1], 0, self.l2, 0])

        # Computes the total transformation matricies to get from frame 0 to frame i
        T_cumulative = [np.eye(4)]
        for i in range(self.num_dof):
            T_cumulative.append(T_cumulative[-1] @ self.T[i])

        # Extracts the translational component to get from frame 0 to the end-effector frame
        d = T_cumulative[-1] @ np.vstack([0, 0, 0, 1])

        # Calculates the jacobian by crossing the distance to the end-effector and the z component of rotation
        for i in range(0, 2):
            T_i = T_cumulative[i]
            z = T_i @ np.vstack([0, 0, 1, 0])
            d1 = T_i @ np.vstack([0, 0, 0, 1])
            r = np.array([d[0] - d1[0], d[1] - d1[1], d[2] - d1[2]]).flatten()
            self.J[i] = np.cross(z[:3].flatten(), r.flatten())


    def calc_numerical_ik(self, EE: EndEffector, tol=0.01, ilimit=500):
        self.theta = [np.radians(20), np.radians(20)]
        for i in range(ilimit):
            self.calc_forward_kinematics(self.theta, radians=True)
            error = [EE.x - self.ee.x, EE.y - self.ee.y]

            if(np.linalg.norm(error) <= tol):
                print("converged")
                break

            self.calculate_jacobian()
            J_inv = np.linalg.pinv(self.J)

            self.theta[0] += (J_inv @ error)[0]
            self.theta[1] += (J_inv @ error)[1]


    def calc_velocity_kinematics(self, vel: list):
        """
        Calculates the velocity kinematics for the robot based on the given velocity input.

        Args:
            vel (list): The velocity vector for the end effector [vx, vy].
        """
        
        T_cumulative = [np.eye(4)]
        for i in range(self.num_dof):
            T_cumulative.append(T_cumulative[-1] @ self.T[i])

        # Extracts the translational component to get from frame 0 to the end-effector frame
        d = T_cumulative[-1] @ np.vstack([0, 0, 0, 1])

        # Calculates the jacobian by crossing the distance to the end-effector and the z component of rotation
        for i in range(0, 2):
            T_i = T_cumulative[i]
            z = T_i @ np.vstack([0, 0, 1, 0])
            d1 = T_i @ np.vstack([0, 0, 0, 1])
            r = np.array([d[0] - d1[0], d[1] - d1[1], d[2] - d1[2]]).flatten()
            self.J[i] = np.cross(z[:3].flatten(), r.flatten())
            
        # Uses psuedoinverse to calculate inverse of jacobian
        # This is done since the jacobian is not square
        J_inv = np.linalg.pinv(self.J)

        theta_dot = np.dot(np.array(vel), J_inv)
        
        # Control cycle time step
        dt = 0.01
        # Calculates next theta values by multiplying angular velocities by time step
        self.theta = self.theta + (theta_dot * dt)
        # Calls forward kinematics with new theta values
        self.calc_forward_kinematics(self.theta, radians=True)

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

class ScaraRobot():
    """
    A class representing a SCARA (Selective Compliance Assembly Robot Arm) robot.
    This class handles the kinematics (forward, inverse, and velocity kinematics) 
    and robot configuration, including joint limits and end-effector calculations.
    """
    
    def __init__(self):
        """
        Initializes the SCARA robot with its geometry, joint variables, and limits.
        Sets up the transformation matrices and robot points.
        """
        # Geometry of the robot (link lengths in meters)
        self.l1 = 0.35  # Base to 1st joint
        self.l2 = 0.18  # 1st joint to 2nd joint
        self.l3 = 0.15  # 2nd joint to 3rd joint
        self.l4 = 0.30  # 3rd joint to 4th joint (tool or end-effector)
        self.l5 = 0.12  # Tool offset

        # Joint variables (angles in radians)
        self.theta = [0.0, 0.0, 0.0]

        # Joint angle limits (min, max) for each joint
        self.theta_limits = [
            [-np.pi, np.pi],
            [-np.pi + 0.261, np.pi - 0.261],
            [0, self.l1 + self.l3 - self.l5]
        ]

        # End-effector (EE) object to store EE position and orientation
        self.ee = EndEffector()

        # Number of degrees of freedom and number of points to store robot configuration
        self.num_dof = 3
        self.num_points = 7
        self.points = [None] * self.num_points

        # Transformation matrices (DH parameters and resulting transformation)
        self.DH = np.zeros((5, 4))  # Denavit-Hartenberg parameters (theta, d, a, alpha)
        self.T = np.zeros((self.num_dof, 4, 4))  # Transformation matrices

        ########################################

        # insert your additional code here

        ########################################

    
    def calc_forward_kinematics(self, theta: list, radians=False):
        """
        Calculate Forward Kinematics (FK) based on the given joint angles.

        Args:
            theta (list): Joint angles (in radians if radians=True, otherwise in degrees).
            radians (bool): Whether the input angles are in radians (default is False).
        """

        if(not radians):
            theta = np.radians(theta)

        self.theta = theta

        # Calculate robot points (e.g., end-effector position)
        self.calc_robot_points()


    def calc_inverse_kinematics(self, EE: EndEffector, soln=0):
        """
        Calculate Inverse Kinematics (IK) based on the input end-effector coordinates.

        Args:
            EE (EndEffector): End-effector object containing desired position (x, y, z).
            soln (int): Solution index (0 or 1), for multiple possible IK solutions.
        """
        elbow_pos = 1
        if(soln == 1):
            elbow_pos = -elbow_pos

        x, y, z = EE.x, EE.y, EE.z
        l1, l2, l3, l4, l5 = self.l1, self.l2, self.l3, self.l4, self.l5

        beta = np.arccos((l2 ** 2 + l4 ** 2 - x ** 2 - y ** 2) / (2 * l2 * l4))
        self.theta[1] = np.pi + (elbow_pos * beta)

        alpha = np.arctan2(l4 * np.sin(self.theta[1]), l2 + l4 * np.cos(self.theta[1]))
        gamma = np.arctan2(y, x)
        self.theta[0] = gamma - alpha

        self.theta[2] = z + l1 + l3 - l5

        # Recalculate Forward Kinematics to update the robot's configuration
        self.calc_forward_kinematics(self.theta, radians=True)


    def calc_velocity_kinematics(self, vel: list):
        """
        Calculate velocity kinematics and update joint velocities.

        Args:
            vel (array): Linear velocities (3D) of the end-effector.
        """
        ########################################

        # insert your code here

        ########################################

        # Recalculate robot points based on updated joint angles
        self.calc_robot_points()
  

    def calc_robot_points(self):
        """
        Calculate the main robot points (links and end-effector position) using the current joint angles.
        Updates the robot's points array and end-effector position.
        """

        self.T[0] = dh_to_matrix([self.theta[0], self.l1, self.l2, 0])
        self.T[1] = dh_to_matrix([self.theta[1], self.l3 - self.l5, self.l4, 0])
        self.T[2] = dh_to_matrix([0, -self.theta[2], 0, 180])

        # Calculate transformation matrices for each joint and end-effector
        self.points[0] = np.array([0, 0, 0, 1])
        self.points[1] = np.array([0, 0, self.l1, 1])
        self.points[2] = self.T[0]@self.points[0]
        self.points[3] = self.points[2] + np.array([0, 0, self.l3, 1])
        self.points[4] = self.T[0]@self.T[1]@self.points[0] + np.array([0, 0, self.l5, 1])
        self.points[5] = self.T[0]@self.T[1]@self.points[0]
        self.points[6] = self.T[0]@self.T[1]@self.T[2]@self.points[0]

        self.EE_axes = self.T[0]@self.T[1]@self.T[2]@np.array([0.075, 0.075, 0.075, 1])
        self.T_ee = self.T[0]@self.T[1]@self.T[2]

        # End-effector (EE) position and axes
        self.ee.x = self.points[-1][0]
        self.ee.y = self.points[-1][1]
        self.ee.z = self.points[-1][2]
        rpy = rotm_to_euler(self.T_ee[:3,:3])
        self.ee.rotx, self.ee.roty, self.ee.rotz = rpy
        
        # EE coordinate axes
        self.EE_axes = np.zeros((3, 3))
        self.EE_axes[0] = self.T_ee[:3,0] * 0.075 + self.points[-1][0:3]
        self.EE_axes[1] = self.T_ee[:3,1] * 0.075 + self.points[-1][0:3]
        self.EE_axes[2] = self.T_ee[:3,2] * 0.075 + self.points[-1][0:3]

class FiveDOFRobot():
    """
    A class to represent a 5-DOF robotic arm with kinematics calculations, including
    forward kinematics, inverse kinematics, velocity kinematics, and Jacobian computation.

    Attributes:
        l1, l2, l3, l4, l5: Link lengths of the robotic arm.
        theta: List of joint angles in radians.
        theta_limits: Joint limits for each joint.
        ee: End-effector object for storing the position and orientation of the end-effector.
        num_dof: Number of degrees of freedom (5 in this case).
        points: List storing the positions of the robot joints.
        DH: Denavit-Hartenberg parameters for each joint.
        T: Transformation matrices for each joint.
    """
    
    def __init__(self):
        """Initialize the robot parameters and joint limits."""
        # Link lengths
        self.l1, self.l2, self.l3, self.l4, self.l5 = 0.30, 0.15, 0.18, 0.15, 0.12
        self.l1 = 15.5 * 0.01
        self.l2 = 9.9 * 0.01
        self.l3 = 9.5 * 0.01
        self.l4 = 5.5 * 0.01
        self.l5 = 10.5 * 0.01
        
        # Joint angles (initialized to zero)
        self.theta = [0, 0, 0, 0, 0]
        
        # Joint limits (in radians)
        self.theta_limits = [
            [-np.pi, np.pi], 
            [-np.pi/3, np.pi], 
            [-np.pi+np.pi/12, np.pi-np.pi/4], 
            [-np.pi+np.pi/12, np.pi-np.pi/12], 
            [-np.pi, np.pi]
        ]

        self.theta_limits = [
            [-(2 * np.pi) / 3, (2 * np.pi) / 3], 
            [-np.pi/2, np.pi/2], 
            [-2 * np.pi / 3, 2 * np.pi / 3],
            [-5 * np.pi / 9, 5 * np.pi / 9], 
            [-np.pi / 2, np.pi / 2], 
        ]
        
        # End-effector object
        self.ee = EndEffector()
        
        # Robot's points
        self.num_dof = 5
        self.points = [None] * (self.num_dof + 1)

        # Denavit-Hartenberg parameters and transformation matrices
        self.DH = np.zeros((5, 4))
        self.T = np.zeros((self.num_dof, 4, 4));
    
        self.J = np.zeros((5, 3))
            
    def calc_forward_kinematics(self, theta: list, radians=False):
        """
        Calculate forward kinematics based on the provided joint angles.

        Args:
            theta: List of joint angles (in degrees or radians).
            radians: Boolean flag to indicate if input angles are in radians.
        """

        # Want to use radians to control the simulation since all the functions are written to work with radian values
        if not radians:
            theta = np.radians(theta)

        # DH Table, derivation shown above
        self.DH = [
            [theta[0], self.l1, 0, -np.pi/2],
            [theta[1] - np.pi/2, 0, self.l2, np.pi],
            [theta[2], 0, self.l3, np.pi],
            [theta[3] + np.pi/2, 0, 0, np.pi/2],
            [theta[4], self.l4 + self.l5, 0, 0],
        ]

        # This vertically stacks the transformation matricies from the DH table
        # self.T represents the transformation required to go from joint i-1 to joint i
        self.T = np.stack(
            [
                dh_to_matrix(self.DH[0]),
                dh_to_matrix(self.DH[1]),
                dh_to_matrix(self.DH[2]),
                dh_to_matrix(self.DH[3]),
                dh_to_matrix(self.DH[4]),
            ],
            axis=0,
        )

        self.theta = theta
        
        # Calculate robot points (positions of joints)
        self.calc_robot_points()


    def check_limits(self, angles):
        for i, angle in enumerate(angles):
            if(angle < self.theta_limits[i][0] and angle > self.theta_limits[i][1]):
                return False
        return True



    def calc_inverse_kinematics(self, EE: EndEffector, soln=0):
        """
        Calculate inverse kinematics to determine the joint angles based on end-effector position.
        
        Args:
            EE: EndEffector object containing desired position and orientation.
            soln: Optional parameter for multiple solutions (not implemented).
        """
        solutions = np.zeros((8, 5))
        R_0_6 = euler_to_rotm((EE.rotx, EE.roty, EE.rotz))
        z_rot = R_0_6[:, 2]
        d5 = self.l4 + self.l5
        pos_j4 = np.array([EE.x, EE.y, EE.z]) - (d5 * z_rot)
        j4_x, j4_y, j4_z = pos_j4[0], pos_j4[1], pos_j4[2]

        R_w = np.sqrt(j4_x ** 2 + j4_y ** 2)
        N = np.sqrt((j4_z - self.l1) ** 2 + R_w ** 2)

        # Could add pi for another sol
        solutions[0:3, 0] = wraptopi(np.arctan2(j4_y, j4_x))
        solutions[4:7, 0] = wraptopi(np.pi + wraptopi(np.arctan2(j4_y, j4_x)))
        
        law_cos = (N ** 2 + self.l2 ** 2 - self.l3 ** 2) / (2 * self.l2 * N)
        mu = np.arccos(np.clip(law_cos, -1, 1))
        lmbda = np.arctan2((j4_z - self.l1), R_w)

        # Could subtract mu for another sol
        solutions[[0, 2, 4, 6], 1]  = wraptopi((np.pi / 2) - lmbda) + mu
        solutions[[1, 3, 5, 7], 1]  = wraptopi((np.pi / 2) - lmbda) - mu
        
        law_cos = (N ** 2 - self.l2 ** 2 - self.l3 ** 2) / (2 * self.l2 * self.l3)
        solutions[[0, 1, 4, 5], 2] = wraptopi(np.arccos(np.clip(law_cos, -1, 1)))
        solutions[[2, 3, 6, 7], 2] = -wraptopi(np.arccos(np.clip(law_cos, -1, 1)))

        for i, solution in enumerate(solutions):
            DH = [
                [solution[0], self.l1, 0, -np.pi/2],
                [solution[1], 0, self.l2, np.pi],
                [solution[2], 0, self.l3, np.pi]
            ]
            
            T_0_1 = dh_to_matrix(DH[0])
            T_1_2 = dh_to_matrix(DH[1])
            T_2_3 = dh_to_matrix(DH[2])
            
            T_0_3 = T_0_1 @ T_1_2 @ T_2_3
            R_0_3 = T_0_3[:3, :3]
            
            R_3_6 = np.transpose(R_0_3) @ R_0_6
            
            solutions[i, 3] = wraptopi((np.pi / 2) + np.arctan2(R_3_6[1, 2], R_3_6[0, 2]))
            solutions[i, 4] = wraptopi(np.arctan2(R_3_6[2, 0], R_3_6[2, 1]))
            
        valid_solutions = []
        for i, solution in enumerate(solutions):
            # Check joint limits first (discard immediately if invalid)
            if not self.check_limits(solution):
                continue
                
            # Calculate achieved position
            target_pos = [EE.x, EE.y, EE.z, EE.rotx, EE.roty, EE.rotz]
            self.calc_forward_kinematics(solution, radians=True)
            achieved_pos = [self.ee.x, self.ee.y, self.ee.z, 
                        self.ee.rotx, self.ee.roty, self.ee.rotz]
            
            # Calculate error
            error = np.linalg.norm(np.array(target_pos) - np.array(achieved_pos))
            
            # Store valid solutions with their error and index
            valid_solutions.append((error, i, solution))

        if not valid_solutions:
            raise ValueError("No valid solutions found within joint limits")

        # Sort by error (ascending - lowest error first)
        valid_solutions.sort()

        # Get the best solution (lowest error)
        best_error, best_index, best_solution = valid_solutions[0]

        # Apply it
        self.calc_forward_kinematics(best_solution, radians=True)

    def check_limits(self, theta_list):
        for i, theta in enumerate(theta_list):
            if(theta < self.theta_limits[i][0] or theta > self.theta_limits[i][1]):
                return False
        return True

    def calc_jacobian(self):
        self.DH = [
            [self.theta[0], self.l1, 0, -np.pi/2],
            [self.theta[1] - np.pi/2, 0, self.l2, np.pi],
            [self.theta[2], 0, self.l3, np.pi],
            [self.theta[3] + np.pi/2, 0, 0, np.pi/2],
            [self.theta[4], self.l4 + self.l5, 0, 0],
        ]

        # This vertically stacks the transformation matricies from the DH table
        # self.T represents the transformation required to go from joint i-1 to joint i
        self.T = np.stack(
            [
                dh_to_matrix(self.DH[0]),
                dh_to_matrix(self.DH[1]),
                dh_to_matrix(self.DH[2]),
                dh_to_matrix(self.DH[3]),
                dh_to_matrix(self.DH[4]),
            ],
            axis=0,
        )

        T_cumulative = [np.eye(4)]
        for i in range(self.num_dof):
            T_cumulative.append(T_cumulative[-1] @ self.T[i])

        # Extracts the translational component to get from frame 0 to the end-effector frame
        d = T_cumulative[-1] @ np.vstack([0, 0, 0, 1])

        # Calculates the jacobian by crossing the distance to the end-effector and the z component of rotation
        for i in range(0, 5):
            T_i = T_cumulative[i]
            z = T_i @ np.vstack([0, 0, 1, 0])
            d1 = T_i @ np.vstack([0, 0, 0, 1])
            r = np.array([d[0] - d1[0], d[1] - d1[1], d[2] - d1[2]]).flatten()
            self.J[i] = np.cross(z[:3].flatten(), r.flatten())
            

    def calc_numerical_ik(self, EE: EndEffector, tol=0.01, ilimit=50):
        """ Calculate numerical inverse kinematics based on input coordinates. """
        
        # Generates random theta valyes between each limit
        for i, limit in enumerate(self.theta_limits):
            min_value = limit[0]
            max_value = limit[1]
            random_value = np.random.uniform(min_value, max_value)
            self.theta[i] = random_value

        for i in range(0, ilimit):
            self.calc_forward_kinematics(self.theta, radians=True)
            error = [EE.x - self.ee.x, EE.y - self.ee.y, EE.z - self.ee.z]
            if(np.linalg.norm(error) <= tol):
                break
            self.calc_jacobian()
            J_inv = np.linalg.pinv(self.J)
            self.theta += error @ J_inv
                

        self.calc_forward_kinematics(self.theta, radians=True)

    
    def calc_velocity_kinematics(self, vel: list):
        """
        Calculate the joint velocities required to achieve the given end-effector velocity.
        
        Args:
            vel: Desired end-effector velocity (3x1 vector).
        """
        

        # Computes the total transformation matricies to get from frame 0 to frame i
        T_cumulative = [np.eye(4)]
        for i in range(self.num_dof):
            T_cumulative.append(T_cumulative[-1] @ self.T[i])

        # Extracts the translational component to get from frame 0 to the end-effector frame
        d = T_cumulative[-1] @ np.vstack([0, 0, 0, 1])

        # Calculates the jacobian by crossing the distance to the end-effector and the z component of rotation
        for i in range(0, 5):
            T_i = T_cumulative[i]
            z = T_i @ np.vstack([0, 0, 1, 0])
            d1 = T_i @ np.vstack([0, 0, 0, 1])
            r = np.array([d[0] - d1[0], d[1] - d1[1], d[2] - d1[2]]).flatten()
            self.J[i] = np.cross(z[:3].flatten(), r.flatten())
            
        # Uses psuedoinverse to calculate inverse of jacobian
        # This is done since the jacobian is not square
        J_inv = np.linalg.pinv(self.J)
        # Multiplies the velocity vector by the inverse jacobian to get angular velocities of each joint
        theta_dot = np.dot(np.array(vel), J_inv)
        
        # Control cycle time step
        dt = 0.01
        # Calculates next theta values by multiplying angular velocities by time step
        self.theta = self.theta + (theta_dot * dt)
        # Calls forward kinematics with new theta values
        self.calc_forward_kinematics(self.theta, radians=True)


    def calc_robot_points(self):
        """ Calculates the main arm points using the current joint angles """

        # Initialize points[0] to the base (origin)
        self.points[0] = np.array([0, 0, 0, 1])

        # Precompute cumulative transformations to avoid redundant calculations
        T_cumulative = [np.eye(4)]
        for i in range(self.num_dof):
            T_cumulative.append(T_cumulative[-1] @ self.T[i])

        # Calculate the robot points by applying the cumulative transformations
        for i in range(1, 6):
            self.points[i] = T_cumulative[i] @ self.points[0]

        # Calculate EE position and rotation
        self.EE_axes = T_cumulative[-1] @ np.array([0.075, 0.075, 0.075, 1])  # End-effector axes
        self.T_ee = T_cumulative[-1]  # Final transformation matrix for EE

        # Set the end effector (EE) position
        self.ee.x, self.ee.y, self.ee.z = self.points[-1][:3]
        
        # Extract and assign the RPY (roll, pitch, yaw) from the rotation matrix
        rpy = rotm_to_euler(self.T_ee[:3, :3])
        self.ee.rotx, self.ee.roty, self.ee.rotz = rpy[0], rpy[1], rpy[2]

        # Calculate the EE axes in space (in the base frame)
        self.EE = [self.ee.x, self.ee.y, self.ee.z]
        self.EE_axes = np.array([self.T_ee[:3, i] * 0.075 + self.points[-1][:3] for i in range(3)])