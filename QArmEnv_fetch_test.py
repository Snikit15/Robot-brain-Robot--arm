'''
The QArmEnv builds on top of gym.env to control the QArm-Simulation
with a RL-Agent.

Author:  Robin Herrmann
Created: 2023.07.12
'''

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.parameter import Parameter
from rclpy.duration import Duration

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState, Image
from control_msgs.msg import JointTrajectoryControllerState

from gazebo_msgs.srv import SpawnEntity
from gazebo_msgs.msg import LinkStates, ModelStates, ContactsState

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

from ament_index_python.packages import get_package_share_directory

import numpy as np

import math
import time
import os
import xacro

from GazeboConnector import GazeboConnector
from JointControllerInterface import JointTrajectoryController
import observation_utils as obs_ut

x_target = 0.45
y_target = 0.3
z_target = 0.05

target_position = np.array([x_target, y_target, z_target])

threshold = 0.05

class QArmEnv(gym.Env):

    def __init__(self):
        TARGET_NAME = 'target'

        JOINT_PUBLISHER = '/joint_trajectory_controller/joint_trajectory'
        JOINT_SUBSCRIBER = '/joint_states'
        MODEL_SUBSCRIBER = '/gazebo/model_states'
        LINK_SUBSCRIBER = '/gazebo/link_states'
        RGB_SUBSCRIBER = '/depth_camera/image_raw'
        DEPTH_SUBSCRIBER = '/depth_camera/depth/image_raw'
        TARGET_COLLISION_SUBSCRIBER = '/'+TARGET_NAME+'/cube_collision'
        QARM_COLLISION_SUBSCRIBER = '/qarm_collision'

        # Time variables
        self.EPISODE_TIMEOUT = 5 # seconds
        self.STEPWAITTIME = 100 # millisase_link

        # List of models collisions are disallowed with
        self.collision_disallowed_models = ['ground_plane::link::collision', '3DPrinterBed::link_0::collision', 
                                            'qarm_v1::bicep_link::bicep_link_collision', 'qarm_v1::yaw_link::yaw_link_collision',
                                            'qarm_v1::base_link::base_link_collision']

        # Defining gym.Env action_space and observation_space
        self.action_space = spaces.Box(
            low = -0.1,
            high = 0.1, 
            shape=(5,), 
            dtype=np.float32,
        )
        # self.observation_space = spaces.Box(
        #     low = np.array([-1, -1, 0, -1, -1, 0, -1, -1, -1, -math.pi, -math.pi]),
        #     high = np.array([1 , 1, 1, 1, 1, 1, 1, 1, 1, math.pi, math.pi]),
        #     dtype=np.float32,
        # )
        #OPTIONAL observation space as dictionary
        self.observation_space = spaces.Dict({
            "TCP": spaces.Box(
                low=np.array([-1, -1, 0]),
                high=np.array([1, 1, 1]),
                dtype=np.float64,
            ),
            "TARGET": spaces.Box(
                low=np.array([-1, -1, 0]),
                high=np.array([1, 1, 1]),
                dtype=np.float64,
            ),
            "RELATIVE": spaces.Box(
                low=np.array([-1, -1, -1]),
                high=np.array([1, 1, 1]),
                dtype=np.float64,
            ),
            "CUBE_target_position": spaces.Box(
                low=np.array([-1, -1, 0]),
                high=np.array([1, 1, 1]),
                dtype=np.float64,
            ),
            "RELATIVE_cube_target_position": spaces.Box(
                low=np.array([-1, -1, -1]),
                high=np.array([1, 1, 1]),
                dtype=np.float64,
            ),
            "GRIPPER": spaces.Box(
                low=np.array([-math.pi, -math.pi]),
                high=np.array([math.pi, math.pi]),
                dtype=np.float64,
            ),
        })

        # Initialize ROS node
        rclpy.init()
        self.node = rclpy.create_node(self.__class__.__name__)
        self.node.set_parameters([Parameter('use_sim_time', value = True)])

        # Initialize Gazebo Connector
        self.gzcon = GazeboConnector(self.node)

        # Create subscriber and publishers
        self._pub = self.node.create_publisher(
            JointTrajectory, JOINT_PUBLISHER, qos_profile=qos_profile_sensor_data)
        self._sub_joint = self.node.create_subscription(
            JointState, JOINT_SUBSCRIBER, self.observation_callback, qos_profile=qos_profile_sensor_data)
        self._sub_model = self.node.create_subscription(
            ModelStates, MODEL_SUBSCRIBER, self.observation_callback, qos_profile=qos_profile_sensor_data)
        self._sub_link = self.node.create_subscription(
            LinkStates, LINK_SUBSCRIBER, self.observation_callback, qos_profile=qos_profile_sensor_data)

        self._sub_cam_rgb = self.node.create_subscription(
            Image, RGB_SUBSCRIBER, self.observation_callback, qos_profile=qos_profile_sensor_data)
        self._sub_cam_depth = self.node.create_subscription(
            Image, DEPTH_SUBSCRIBER, self.observation_callback, qos_profile=qos_profile_sensor_data)

        self._sub_target_collision = self.node.create_subscription(
            ContactsState, TARGET_COLLISION_SUBSCRIBER, self.target_collision_observation_callback, qos_profile=qos_profile_sensor_data)
        self._sub_qarm_collision = self.node.create_subscription(
            ContactsState, QARM_COLLISION_SUBSCRIBER, self.qarm_collision_observation_callback, qos_profile=qos_profile_sensor_data)

        self._sub_count = 7
        self._sub_count_additional = 7-1 # count of collision publishers

        # Initialize variables
        self._msg_observation = "None"
        self._msg_joint_state= "None"
        self._msg_model_state = "None"
        self._msg_link_state = "None"

        self._msg_camera_rgb = "None"
        self._msg_camera_depth = "None"

        self._msg_target_collision = ContactsState()
        self._msg_qarm_collision = ContactsState()

        self.last_joints_complete = {"name": [], "position": [0,0,0,0,0,0,0,0]}

        self.last_reset_clock = 0

        # Spawn Target
        xacro_file = os.path.join(get_package_share_directory(
            'qarm_v1'), 'urdf', 'cube.urdf.xacro')    
        xml_cube = xacro.process_file(xacro_file).toxml()#.replace('"', '\\"')
        self.gzcon.spawn_entity(TARGET_NAME, xml_cube, [0.4, 0.2, 0.11])

        # Initial observation
        self.gzcon.unpause()
        time.sleep(5) # important to wait for ros2_control init
        init_observation = self.take_observation()
        self.gzcon.pause()

        self._last_distance = np.linalg.norm(init_observation["RELATIVE"][0:3])

        # Initilize controller connection
        self.arm_control = JointTrajectoryController(
            'joint_arm_position_controller', ['base_yaw_joint', 'yaw_bicep_joint', 
            'bicep_forearm_joint', 'forearm_endeffector_joint'], 'position')

        self.gripper_control = JointTrajectoryController(
            'joint_gripper_position_controller', ['a1_joint', 'a2_joint', 
            'b1_joint', 'b2_joint'], 'position')
    
        # Initial arm position
        self.arm_control.command([0, 0, 0, 0], 0)
        self.gripper_control.command([0, 0, 0, 0], 0)

    def observation_callback(self, message):
        if type(message) == ModelStates:
            self._msg_model_state = message
        elif type(message) == LinkStates:
            self._msg_link_state = message
        elif type(message) == JointState:
            self._msg_joint_state= message
        elif type(message) == Image:
            if message.encoding == "rgb8":
                self._msg_camera_rgb = message
            else:
                self._msg_camera_depth = message
        else:
            self._msg_observation = message
            # self.node.get_logger().error("Received unexpected message type in observation_callback()")
    
    def target_collision_observation_callback(self, message):
        if type(message) == ContactsState:
            self._msg_target_collision = message
        else:
            return
            # self.node.get_logger().error("Received wrong message type in target_collision_observation_callback()")

    def qarm_collision_observation_callback(self, message):
        if type(message) == ContactsState:
            self._msg_qarm_collision = message
        else:
            return
            # self.node.get_logger().error("Received wrong message type in qarm_collision_observation_callback()")
    
    def check_forbidden_collisions(self):
        for collision_state in self._msg_qarm_collision.states:
            if any(disallowed in collision_state.collision1_name or 
                disallowed in collision_state.collision2_name
                for disallowed in self.collision_disallowed_models):
                return True
        return False
    
    def random_cube_position(self):
        # random position
        y = np.random.uniform(-0.45, 0.45)
        x = np.random.uniform(0.25, 0.45)
        z = 0.11
        return [x, y, z]


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
        obs_relative = obs_ut.process_obs_relative(obs_tcp, obs_target)
        obs_relative_cube_target_position = obs_ut.process_obs_relative(obs_target, target_position)
        print(obs_gripper)
        # Dict observation
        observation = {
            "TCP": obs_tcp,
            "TARGET": obs_target,
            "RELATIVE": obs_relative,
            "CUBE_target_position":target_position,
            "RELATIVE_cube_target_position": obs_relative_cube_target_position,
            "GRIPPER": obs_gripper,
        }

        # self.node.get_logger().info("QArmEnv.step(); Command " + str(observation))

        return observation


    def step(self, action):

        self.ros_clock = self.node.get_clock().now().nanoseconds

        # make the move according to the Agent
        arm_command = self.last_joints_complete['position'][0:4] + action[0:4]
        gripper_command = self.last_joints_complete['position'][4:8] + np.array([action[4], action[4], action[4], action[4]])
        self.arm_control.command(arm_command.tolist(), self.STEPWAITTIME/1000)
        self.gripper_control.command(gripper_command.tolist(), self.STEPWAITTIME/1000)

        # wait until the move is finisched
        self._clock_last_step = self.node.get_clock().now().nanoseconds
        while self.node.get_clock().now().nanoseconds - self._clock_last_step <= self.STEPWAITTIME*1000000:
            rclpy.spin_once(self.node)

        observation = self.take_observation()

        cube_position = observation["TARGET"]
        tcp_position = observation["TCP"]
        bicep_winkel = observation["GRIPPER"][1]
        cube_distance = np.linalg.norm(observation["RELATIVE_cube_target_position"])
        tcp_cube_distance = np.linalg.norm(observation["RELATIVE"])

        # print("cube_position",cube_position)
        # print("TCP",observation["TCP"])
        #print(cube_distance)

        #make the bicep not go to too negative winkel.
        if bicep_winkel<-0.3:
            bicep_reward = -10
        else:
            bicep_reward = 0
        
        target_min = 0.05
        target_max = 0.1
        position = tcp_position[2]

        if target_min <= position <= target_max:
            # the heigt of the grip is good
            TCP_reward = 0.1
        else:
            # the heigt of the grip is not good
            if position < target_min:
                punishment = -10
            else:
                punishment = -(position-0.08)*10
            TCP_reward = punishment
        # print("TCP_position_z ",position)
        
        current_cube_position = cube_position
        position_change = np.linalg.norm(current_cube_position - self.previous_cube_position)

        # check if cube moves
        position_change_threshold = 0.01  # adjust the threshold
        if position_change > position_change_threshold:
            # cube moves
            position_change_reward = 5  # reward
        else:
            # not move
            position_change_reward = -0.5  # punishment
        # actalize the cube position
        self.previous_cube_position = current_cube_position
            
        # the new distance between TCP and cube atfer the move
        new_distance = np.linalg.norm(observation["RELATIVE"])
        if new_distance < self.previous_distance:
            # if the distance is closer
            grip_moved_reward = 0.1
        elif new_distance >= self.previous_distance:
            # if the distance is further
            grip_moved_reward = -1
        self.previous_distance = new_distance  # refresh the distance

        # check the unacceptable collision
        if self.check_forbidden_collisions():
            collision_reward = -50 
            # truncated = True    # truncate this episode
            # # info = {'reason': 'forbidden_collision'}
            # return (observation, reward, False, truncated, {})
        else:
            collision_reward = 1

        reward = -0.1 + 10*(-tcp_cube_distance) - 50*cube_distance + 2*bicep_reward +5*TCP_reward + grip_moved_reward + 5*position_change_reward + 1*collision_reward
        
        info = {'reason': 'max_step reached'}
        
        if np.linalg.norm(cube_position - [0, 0, 0]) > 0.7:
            return (observation, reward, False, True, info)

        # check if it is terminated(if the cube get the target position)
        terminated = False
        if cube_distance <= threshold:  
            terminated = True
            reward += 100  # reward
            info = {'reason': 'task finisched'}
            return (observation, reward, terminated, False, info)
        
        print("reward ",reward)

        return (observation, reward, terminated, False, info)

    
    def reset(self, seed=None, options={}):
        self.gzcon.unpause()
        self.gzcon.reset_world()

        self.gzcon.delete_entity('target')

        time.sleep(0.2)

        cube_position = self.random_cube_position()
        xacro_file = os.path.join(get_package_share_directory(
        'qarm_v1'), 'urdf', 'cube.urdf.xacro')    
        xml_cube = xacro.process_file(xacro_file).toxml()
        self.gzcon.spawn_entity('target', xml_cube, coords = cube_position)

        time.sleep(0.3)

        # ramdomly spawn the place of joint
        arm_initial_positions = np.random.uniform(low=[-1.0, -1.0, -1.0, -1.0], high=[1.0, 1.0, 1.0, 1.0], size=4)
        self.arm_control.command(arm_initial_positions.tolist(), 0)

        self.gripper_control.command([0, 0, 0, 0], 0)

        time.sleep(0.2)

        state = self.take_observation()
        self.last_reset_clock = self.node.get_clock().now().nanoseconds
        #self._last_distance = np.linalg.norm(state["RELATIVE"][0:3])

        info = {}
        observation = {
            "TCP": state["TCP"],
            "TARGET": state["TARGET"],
            "RELATIVE": state["RELATIVE"],
            "CUBE_target_position":state["CUBE_target_position"],
            "RELATIVE_cube_target_position":state["RELATIVE_cube_target_position"],
            "GRIPPER": state["GRIPPER"],

        }

        self.previous_distance = np.inf  # init. the distance between TCP and cube
        # self.previous_cube_distance = np.inf  # init. the distance between cube and the target position of the cube
        self.previous_cube_position = np.array([0, 0, 0]) 
        return observation, info