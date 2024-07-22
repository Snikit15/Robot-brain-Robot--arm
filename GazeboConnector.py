'''
The GazeboConnector provides functions to call ROS-Services to
pause, unpause and reset the Gazebo simulation. Also exposes
spawn_entity service.

Author:  Robin Herrmann
Created: 2023.07.12
'''

import rclpy
from rclpy.action import ActionClient
from rclpy.parameter import Parameter
from std_srvs.srv import Empty
from gazebo_msgs.srv import SpawnEntity
from gazebo_msgs.srv import DeleteEntity
from geometry_msgs.msg import Pose, Point
import time
import os
from ament_index_python.packages import get_package_share_directory
import xacro


class GazeboConnector():

    def __init__(self, node):
        self.node = node

        # Initialize service clients
        self._pause_client = node.create_client(Empty, "/pause_physics")
        self._unpause_client = node.create_client(Empty, "/unpause_physics")
        self._reset_world_client = node.create_client(Empty, "/reset_world")
        self._spawn_entity = node.create_client(SpawnEntity, "/spawn_entity")
        self._delete_entity = node.create_client(DeleteEntity, "/delete_entity")

    def pause(self):
        # Function sends pause call
        self._pause_client.call_async(Empty.Request())
        return

    def unpause(self):
        # Function sends unpause call
        self._unpause_client.call_async(Empty.Request())
        return

    def reset_world(self):
        # Function sends reset_world call and waits
        target_future = self._reset_world_client.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self.node, target_future)
        while not self._reset_world_client.wait_for_service(timeout_sec=1.0):
            if not rclpy.ok():
                self.node.get_logger().error('Interruped while waiting for the server _reset_world_client.')
                return
            else:
                self.node.get_logger().info('Server _reset_world_client not available, waiting again...')

    def spawn_entity(self, name, xml, coords = [0,0,0]):
        if not self._spawn_entity.service_is_ready():
            self.node.get_logger().info('Waiting for spawn_entity service...')
            self._spawn_entity.wait_for_service()
            self.node.get_logger().info('spawn_entity service is available.')

        request = SpawnEntity.Request()
        request.name = name
        request.xml = xml
        request.robot_namespace = '/' + name

        initial_pose = Pose()
        initial_point = Point(x=coords[0], y=coords[1], z=coords[2])
        initial_pose.position = initial_point

        request.initial_pose = initial_pose

        future = self._spawn_entity.call_async(request)
        rclpy.spin_until_future_complete(self.node, future)

        if future.result() is not None:
            self.node.get_logger().info(f'Entity {name} spawned successfully.')
        else:
            self.node.get_logger().error('Failed to spawn entity.')

    def delete_entity(self, name):
        if not self._delete_entity.service_is_ready():
            self.node.get_logger().info('Waiting for delete_entity service...')
            self._delete_entity.wait_for_service()
            self.node.get_logger().info('delete_entity service is available.')

        request = DeleteEntity.Request()
        request.name = name

        future = self._delete_entity.call_async(request)
        rclpy.spin_until_future_complete(self.node, future)

        if future.result() is not None and future.result().success:
            self.node.get_logger().info(f'Entity {name} deleted successfully.')
        else:
            self.node.get_logger().error(f'Failed to delete entity {name}.')

if __name__ == "__main__":
    # Tests
    rclpy.init()

    node = rclpy.create_node('Test')
    node.set_parameters([Parameter('use_sim_time', value = 1)])

    gzcon = GazeboConnector(node)

    # gzcon.unpause()
    # time.sleep(5)
    # gzcon.pause()

    gzcon.reset_world()

    xacro_file = os.path.join(get_package_share_directory(
        'qarm_v1'), 'urdf', 'cube.urdf.xacro')    
    xml_cube = xacro.process_file(xacro_file).toxml()#.replace('"', '\\"')

    gzcon.spawn_entity('target', xml_cube, [0.4, 0.4, 0.1])