import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.parameter import Parameter
from trajectory_msgs.msg import JointTrajectory
from sensor_msgs.msg import JointState, Image
from gazebo_msgs.msg import LinkStates, ModelStates, ContactsState
import gymnasium as gym
from gymnasium import spaces
from ament_index_python.packages import get_package_share_directory
import numpy as np
import math
import time
import os
import xacro
from GazeboConnector import GazeboConnector
from JointControllerInterface import JointTrajectoryController
import observation_utils as obs_ut


class QArmEnv(gym.Env):

    def __init__(self):

        self.init_config()
        self.setup_ros_communications()
        self.init_spaces()
        self.init_state_variables()
        self.init_gazebo_and_controllers()  # This now includes GazeboConnector and controller connections
        #self.init_gazebo_entities()

    def init_config(self):
        """Initialize configuration parameters."""
        self.target_name = 'target'
        # self.target_position = np.array([0.45, 0.3, 0.05])
        self.STEPWAITTIME = 100 # millisase_link

        # List of models collisions are disallowed with
        self.collision_disallowed_models = [
            'ground_plane::link::collision', '3DPrinterBed::link_0::collision', 
            'qarm_v1::bicep_link::bicep_link_collision', 'qarm_v1::yaw_link::yaw_link_collision',
            'qarm_v1::base_link::base_link_collision'
        ]
    
    def init_state_variables(self):
        """Initialize state variables for subscriptions and messages."""
        self._sub_count = 7
        self._sub_count_additional = 7-1  # count of collision publishers

        self._msg_observation = "None"
        self._msg_joint_state = "None"
        self._msg_model_state = "None"
        self._msg_link_state = "None"
        self._msg_camera_rgb = "None"
        self._msg_camera_depth = "None"

        self._msg_target_collision = ContactsState()
        self._msg_qarm_collision = ContactsState()

        self.last_joints_complete = {"name": [], "position": [0, 0, 0, 0, 0, 0, 0, 0]}
        self.last_reset_clock = 0

    def init_gazebo_entities(self):
        """Spawn entities in Gazebo and initialize observations."""
        self.spawn_target([0.4, 0.2, 0.05])
        self.gzcon.unpause()
        time.sleep(5)  # Important to wait for ros2_control init
        init_observation = self.take_observation()
        self.gzcon.pause()
        self._last_distance = np.linalg.norm(init_observation["RELATIVE"][0:3])

    def init_gazebo_and_controllers(self):
        """Initialize Gazebo connection and controllers."""
        self.gzcon = GazeboConnector(self.node)
        self.init_controller_connections()

    def init_controller_connections(self):
        """Initialize connections with joint controllers and set initial positions."""
        self.arm_control = JointTrajectoryController(
            'joint_arm_position_controller', ['base_yaw_joint', 'yaw_bicep_joint', 
            'bicep_forearm_joint', 'forearm_endeffector_joint'], 'position')
        self.gripper_control = JointTrajectoryController(
            'joint_gripper_position_controller', ['a1_joint', 'a2_joint', 
            'b1_joint', 'b2_joint'], 'position')

        # Initial arm and gripper positions
        self.arm_control.command([0, 0, 0, 0], 0)
        self.gripper_control.command([0, 0, 0, 0], 0)

    def setup_ros_communications(self):
        """Setup all ROS publishers and subscribers."""
        rclpy.init()
        self.node = rclpy.create_node(self.__class__.__name__)
        self.node.set_parameters([Parameter('use_sim_time', value=True)])
        self.create_publishers()
        self.create_subscribers()
    
    def create_publishers(self):
        """Create ROS publishers for the environment."""
        JOINT_TARJECTORY = '/joint_trajectory_controller/joint_trajectory'
        self._pub = self.node.create_publisher(JointTrajectory, JOINT_TARJECTORY, qos_profile_sensor_data)

    def create_subscribers(self):
        """Create ROS subscribers for the environment."""
        JOINT_SUBSCRIBER = '/joint_states'
        MODEL_SUBSCRIBER = '/gazebo/model_states'
        LINK_SUBSCRIBER = '/gazebo/link_states'
        RGB_SUBSCRIBER = '/depth_camera/image_raw'
        DEPTH_SUBSCRIBER = '/depth_camera/depth/image_raw'
        TARGET_COLLISION_SUBSCRIBER = '/'+self.target_name+'/cube_collision'
        QARM_COLLISION_SUBSCRIBER = '/qarm_collision'

        self._sub_joint = self.node.create_subscription(
            JointState, JOINT_SUBSCRIBER, self.observation_callback, qos_profile_sensor_data)
        self._sub_model = self.node.create_subscription(
            ModelStates, MODEL_SUBSCRIBER, self.observation_callback, qos_profile_sensor_data)
        self._sub_link = self.node.create_subscription(
            LinkStates, LINK_SUBSCRIBER, self.observation_callback, qos_profile_sensor_data)
        self._sub_cam_rgb = self.node.create_subscription(
            Image, RGB_SUBSCRIBER, self.observation_callback, qos_profile_sensor_data)
        self._sub_cam_depth = self.node.create_subscription(
            Image, DEPTH_SUBSCRIBER, self.observation_callback, qos_profile_sensor_data)
        self._sub_target_collision = self.node.create_subscription(
            ContactsState, TARGET_COLLISION_SUBSCRIBER, self.target_collision_observation_callback, qos_profile=qos_profile_sensor_data)
        self._sub_qarm_collision = self.node.create_subscription(
            ContactsState, QARM_COLLISION_SUBSCRIBER, self.qarm_collision_observation_callback, qos_profile=qos_profile_sensor_data)
    
    def init_spaces(self):
        """Define action and observation spaces."""
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(5,), dtype=np.float32)
        # observation space as dictionary
        self.observation_space = spaces.Dict({
            "TCP": spaces.Box(
                low=np.array([-1, -1, 0]),
                high=np.array([1, 1, 1]),
                dtype=np.float64,
            ),
            "CUBE": spaces.Box(
                low=np.array([-1, -1, 0]),
                high=np.array([1, 1, 1]),
                dtype=np.float64,
            ),
            # "RELATIVE": spaces.Box(
            #     low=np.array([-1, -1, -1]),
            #     high=np.array([1, 1, 1]),
            #     dtype=np.float64,
            # ),
            "TARGET_POSITION": spaces.Box(
                low=np.array([-1, -1, 0]),
                high=np.array([1, 1, 1]),
                dtype=np.float64,
            ),
            # "RELATIVE_cube_target_position": spaces.Box(
            #     low=np.array([-1, -1, -1]),
            #     high=np.array([1, 1, 1]),
            #     dtype=np.float64,
            # ),
            "STATE_JOINT_GRIPPER": spaces.Box(
                low=np.array([-math.pi, -math.pi]),
                high=np.array([math.pi, math.pi]),
                dtype=np.float64,
            ),
        })


    def observation_callback(self, message):
        try:
            if isinstance(message, ModelStates):
                self._msg_model_state = message
            elif isinstance(message, LinkStates):
                self._msg_link_state = message
            elif isinstance(message, JointState):
                self._msg_joint_state = message
            elif isinstance(message, Image):
                if message.encoding == "rgb8":
                    self._msg_camera_rgb = message
                else:
                    self._msg_camera_depth = message
            else:
                self.node.get_logger().error(f"Unexpected message type: {type(message)}")
        except Exception as e:
            self.node.get_logger().error(f"Error processing message: {str(e)}")
    
    def target_collision_observation_callback(self, message):
        """Handle incoming messages related to target collisions."""
        if isinstance(message, ContactsState):
            self._msg_target_collision = message
        else:
            self.node.get_logger().error("Received wrong message type for target collision.")

    def qarm_collision_observation_callback(self, message):
        """Handle incoming messages related to QArm collisions."""
        if isinstance(message, ContactsState):
            self._msg_qarm_collision = message
        else:
            self.node.get_logger().error("Received wrong message type for QArm collision.")

    def check_forbidden_collisions(self):
        """Check if any of the recorded collisions are not allowed."""
        for collision_state in self._msg_qarm_collision.states:
            if any(disallowed in collision_state.collision1_name or disallowed in collision_state.collision2_name
                for disallowed in self.collision_disallowed_models):
                return True
        return False
    
    def random_cube_position(self):
        # random position
        y = np.random.uniform(-0.4, 0.4)
        x = np.random.uniform(-0.5, 0.5)
        z = 0.11
        cube_position = [x,y,z]

        # Generate randomized target locations
        target_x = np.random.uniform(0.3, 0.5)
        target_y = np.random.uniform(-0.4, 0.4)
        target_z = 0.05 #height
        target_position = np.array([target_x, target_y, target_z])

        return cube_position, target_position


    def take_observation(self):
        # Spin node as often as there are messages to read
        for i in range(self._sub_count):
            rclpy.spin_once(self.node)

        # Take snapshot of observation
        obs_msg_joints = self._msg_joint_state
        obs_msg_models = self._msg_model_state
        obs_msg_links = self._msg_link_state

        # optional: additionally process camera data from self._msg_camera_depth here

        obs_gripper, self.last_joints_complete = obs_ut.process_obs_msg_joints(obs_msg_joints)
        obs_target = obs_ut.process_obs_msg_models(obs_msg_models)
        obs_tcp = obs_ut.process_obs_msg_links(obs_msg_links)
        # obs_relative = obs_ut.process_obs_relative(obs_tcp, obs_target)
        # obs_relative_cube_target_position = obs_ut.process_obs_relative(obs_target, self.target_position)
        print(obs_gripper)
        # Dict observation
        observation = {
            "TCP": obs_tcp,
            "CUBE": obs_target,
            # "RELATIVE": obs_relative,
            "TARGET_POSITION":self.target_position,
            # "RELATIVE_CUBE_TARGET": obs_relative_cube_target_position,
            "STATE_JOINT_GRIPPER": obs_gripper,  #also include all state of joints besides state of grip
        }

        # self.node.get_logger().info("QArmEnv.step(); Command " + str(observation))

        return observation
    
    def reset(self, seed=None, options={}):
        self.gzcon.unpause()
        self.gzcon.reset_world()

        self.gzcon.delete_entity('target')

        time.sleep(0.1)

        # ramdomly spawn the place of joint
        arm_initial_positions = np.random.uniform(low=[-1.0, -1.0, -1.0, -1.0], high=[1.0, 1.0, 1.0, 1.0], size=4)
        self.arm_control.command(arm_initial_positions.tolist(), 0)

        self.gripper_control.command([0, 0, 0, 0], 0)

        time.sleep(0.2)

        cube_position, self.target_position = self.random_cube_position()
        xacro_file = os.path.join(get_package_share_directory(
        'qarm_v1'), 'urdf', 'cube.urdf.xacro')    
        xml_cube = xacro.process_file(xacro_file).toxml()
        self.gzcon.spawn_entity('target', xml_cube, coords = cube_position)

        time.sleep(0.2)

        state = self.take_observation()
        self.last_reset_clock = self.node.get_clock().now().nanoseconds
        #self._last_distance = np.linalg.norm(state["RELATIVE"][0:3])

        info = {}
        observation = {
            "TCP": state["TCP"],
            "CUBE": state["CUBE"],
            # "RELATIVE": state["RELATIVE"],
            "TARGET_POSITION":state["TARGET_POSITION"],
            # "RELATIVE_CUBE_TARGET":state["RELATIVE_CUBE_TARGET"],
            "STATE_JOINT_GRIPPER": state["STATE_JOINT_GRIPPER"],
        }

        self.previous_distance = np.inf  # init. the distance between TCP and cube
        self.previous_cube_distance = np.inf  # init. the distance between cube and the target position of the cube
        self.previous_cube_position = np.array([0, 0, 0]) 
        return observation, info

    def apply_action(self, arm_command, gripper_command):
        """Applies the specified arm and gripper commands."""
        self.arm_control.command(arm_command.tolist(), self.STEPWAITTIME / 1000)
        self.gripper_control.command(gripper_command.tolist(), self.STEPWAITTIME / 1000)

    def wait_for_action_completion(self):
        """Waits for the specified duration for the action to complete."""
        start_time = self.node.get_clock().now().nanoseconds
        while self.node.get_clock().now().nanoseconds - start_time <= self.STEPWAITTIME * 1000000:
            rclpy.spin_once(self.node)

    def compute_reward(self, observation):
        """Computes the reward based on the current and previous state observations."""
        # Extract relevant positions from observation
        # TODO: base_orientation should be directional to the cube

        cube_position = observation["CUBE"]
        tcp_position = observation["TCP"]
        target_position = observation["TARGET_POSITION"]
        tcp_cube_distance = np.linalg.norm(cube_position - tcp_position)
        cube_target_distance = np.linalg.norm(target_position - cube_position)
        
        reward = -0.01 - 0.1 * tcp_cube_distance - 0.2 * cube_target_distance 
        return reward


    def step(self, action, threshold=0.03):
        """Performs a step in the environment using the given action."""
        arm_command = self.last_joints_complete['position'][0:4] + action[0:4]
        # [4,5,6,7] are the dimension for 4 parts of grip, but is 4* necessary?
        gripper_command = self.last_joints_complete['position'][4:8] + np.array([action[4]] * 4)

        self.apply_action(arm_command, gripper_command)
        self.wait_for_action_completion()

        observation = self.take_observation()
        reward = self.compute_reward(observation)

        # Check for termination conditions
        terminated = np.linalg.norm(observation["TARGET_POSITION"] - observation["CUBE"]) <= threshold
        truncated = self.check_forbidden_collisions()

        if terminated:
            reward += 200  # Bonus for successful completion

        if truncated:
            reward -= 200  # Penalty for forbidden collision
            info = {'reason': 'forbidden_collision'}
            return observation, reward, False, truncated, info


        info = {"reward_components": {"total": reward}}
        return observation, reward, terminated, False, info
    
    def clear_entities(self):
        """Delete any existing entities such as targets."""
        self.gzcon.delete_entity('target')

    def initialize_arm_and_gripper(self):
        """Randomly set initial positions for the arm and gripper."""
        arm_initial_positions = np.random.uniform(low=[-1.0, -1.0, -1.0, -1.0], high=[1.0, 1.0, 1.0, 1.0], size=4)
        self.arm_control.command(arm_initial_positions.tolist(), 0)
        self.gripper_control.command([0, 0, 0, 0], 0)

    def spawn_target(self, position=None):
        """Spawn a target entity at the specified or random position."""
        if position is None:
            # If no specific position is given, generate a random position
            position, self.target_position = self.random_cube_position()

        xacro_file = os.path.join(get_package_share_directory('qarm_v1'), 'urdf', 'cube.urdf.xacro')
        xml_cube = xacro.process_file(xacro_file).toxml()
        self.gzcon.spawn_entity('target', xml_cube, coords=position)

    def reset_tracking_variables(self, initial_state):
        """Reset and initialize tracking variables for the new episode."""
        self.previous_distance = np.inf  # init the distance between TCP and cube
        self.previous_cube_distance = np.inf  # init the distance between cube and the target position
        self.previous_cube_position = initial_state["CUBE"]

    def ensure_target_exists(self):
        """Ensure the 'target' model is loaded and present in the model states."""
        timeout = time.time() + 10  # 10 seconds from now
        while time.time() < timeout:
            try:
                if "target" in self.node.get_model_state().name:
                    return True
            except Exception as e:
                print(f"Waiting for target to be available: {str(e)}")
            time.sleep(0.1)
        return False

    def reset(self, seed=None, options={}):
        # Unpause the simulation and reset the world
        self.gzcon.unpause()
        self.gzcon.reset_world()

        # Clear existing entities
        self.clear_entities()
        time.sleep(2)

        # Set initial positions for arm and gripper
        self.initialize_arm_and_gripper()

        # Spawn a new target entity
        self.spawn_target() 

        self.ensure_target_exists()
        # Capture the initial state after all changes
        observation = self.take_observation()

        # Reset tracking variables
        self.reset_tracking_variables(observation)

        # Optionally return any additional info
        info = {
            "initial_positions": {
                "arm": observation["TCP"],
                "gripper": observation["STATE_JOINT_GRIPPER"]
            }
        }

        return observation, info