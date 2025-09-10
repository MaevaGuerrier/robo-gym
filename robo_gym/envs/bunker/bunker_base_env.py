#!/usr/bin/env python3
import math
import copy
import numpy as np
import gymnasium as gym
from typing import  Optional, Dict, Any, Tuple
from robo_gym.utils import interbotix_utils
from robo_gym.utils.exceptions import InvalidStateError, RobotServerError, InvalidActionError
import robo_gym_server_modules.robot_server.client as rs_client
from robo_gym.utils.camera import RoboGymCamera
from robo_gym_server_modules.robot_server.grpc_msgs.python import robot_server_pb2
from robo_gym.envs.simulation_wrapper import Simulation

IMAGE_SHAPE = [120, 160, 3]

class BunkerRBaseEnv(gym.Env):
    """Interbotix rover base environment.

    Args:
        rs_address (str): Robot Server address. Formatted as 'ip:port'. Defaults to None.
        robot_model (str): determines which interbotix rover model will be used in the environment. Default to 'locobot_wx250s'.

    Attributes:
        interbotix (:obj:): Robot utilities object.
        client (:obj:str): Robot Server client.
        real_robot (bool): True if the environment is controlling a real robot.

    """
    real_robot = False
    max_episode_steps = 100

    def __init__(self, rs_address=None, robot_model='bunker', rs_state_to_info=True, with_camera=True, context_size=1, **kwargs):

        self.elapsed_steps = 0
        self.rs_state = None
        self.rs_state_to_info = rs_state_to_info
        self.camera = with_camera
        self.context_size = context_size

        self.rs_state_to_info = rs_state_to_info
                
        if self.camera:
            self.camera_config = RoboGymCamera(name='camera', image_shape=IMAGE_SHAPE,
                                        image_mode='temporal', context_size=self.context_size, num_cameras=1)
        
        self.base_pose_list = ['base_position_x', 'base_position_y', 'base_position_z', 'base_orientation_x', 
                               'base_orientation_y', 'base_orientation_z', 'base_orientation_w']
        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()




        # Connect to Robot Server
        if rs_address:
            self.client = rs_client.Client(rs_address)
        else:
            print("WARNING: No IP and Port passed. Simulation will not be started")
            print("WARNING: Use this only to get environment shape")
            
    def _set_initial_robot_server_state(self, rs_state) -> robot_server_pb2.State:
        string_params = {}
        float_params = {}
        state = {}

        state_msg = robot_server_pb2.State(state=state, float_params=float_params,
                                           string_params=string_params, state_dict=rs_state)
        return state_msg
    
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:

        """Environment reset.

        Args:
            joint_positions (list[6] or np.array[6]): robot joint positions in radians. Order is defined by 
        
        Returns:
            np.array: Environment state.

        """
        super().reset(seed=seed)

        if options is None:
            options = {}


        self.elapsed_steps = 0

        state_len = self.observation_space['state'].shape[0]
        state={}
        state['state'] = np.zeros(state_len)

        # Initialize environment state
        

        rs_state = dict.fromkeys(self.get_robot_server_composition(), 0.0)

        # Set initial robot joint goals
        base_velocity = [0, 0]

        # Set initial state of the Robot Server
        state_msg = self._set_initial_robot_server_state(rs_state)
        if not self.client.set_state_msg(state_msg):
            raise RobotServerError("set_state")

        # Get Robot Server state
        rs_state_all = self.client.get_state_msg()
        rs_state = rs_state_all.state_dict

        # Check if the length and keys of the Robot Server state received is correct
        self._check_rs_state_keys(rs_state)

        # Convert the initial state from Robot Server format to environment format
        state['state'] = self._robot_server_state_to_env_state(rs_state)

        if not self.camera and not self.observation_space['state'].contains(state['state']):
            raise InvalidStateError()


        self.rs_state = rs_state
        if self.camera:
             state['camera'] = self.camera_config.process_camera_images(rs_state_all.string_params)

        return state, {}
    
    def reward(self, rs_state, action) -> Tuple[float, bool, dict]:
        done = False
        info = {}

        if self.elapsed_steps >= self.max_episode_steps:
            done = True
            info['final_status'] = 'success'

        return 0, done, info
    
    def env_action_to_rs_action(self, action) -> np.ndarray:
        """Convert environment action to Robot Server action"""
        rs_action = copy.deepcopy(action)
        rs_action_all = rs_action
    
        return rs_action_all        

    def step(self, action) -> Tuple[np.array, float, bool, dict]:
        # action should be a list/array with the following joint velocities, depending on 
        # dof some arm joints can be ommitted.
        #  [base_velocity_linear_x, base_velocity_angular_z]
        
        if type(action) == list:
            action = np.array(action)

        rs_action = []
        self.elapsed_steps += 1
        state = {}

        action = action.astype(np.float32)

        # Check if the action is contained in the action space
        if not self.action_space.contains(action):
            raise InvalidActionError()

        # Convert environment action to robot server action
        rs_action = self.env_action_to_rs_action(action)

        # Send action to Robot Server and get state
        rs_state_all = self.client.send_action_get_state(rs_action.tolist())
        rs_state = rs_state_all.state_dict
        self._check_rs_state_keys(rs_state)

        # Convert the state from Robot Server format to environment format
        state['state'] = self._robot_server_state_to_env_state(rs_state)

        if self.camera:
             state['camera'] = self.camera_config.process_camera_images(rs_state_all.string_params)
            
        if not self.camera and not self.observation_space['state'].contains(state['state']):
            raise InvalidStateError()

        # Check if the environment state is contained in the observation space
        # if not self.observation_space.contains(state):
        #     raise InvalidStateError()

        self.rs_state = rs_state

        # Assign reward
        reward = 0
        done = False
        reward, done, info = self.reward(rs_state=rs_state, action=action)
        if self.rs_state_to_info:
            info['rs_state'] = self.rs_state

        return state, reward, done, False, info

    def get_rs_state(self):
        return self.rs_state

    def render(self):
        pass
    
    def get_robot_server_composition(self) -> list:

        rs_state_keys = self.base_pose_list 

        return rs_state_keys
    
        
    def _check_rs_state_keys(self, rs_state) -> None:
        keys = self.get_robot_server_composition()
        
        if not len(keys) == len(rs_state.keys()):
            raise InvalidStateError("Robot Server state keys to not match. Different lengths.")

        for key in keys:
            if key not in rs_state.keys():
                raise InvalidStateError("Robot Server state keys to not match")
            
    def _robot_server_state_to_env_state(self, rs_state) -> np.ndarray:
        """Transform state from Robot Server to environment format.

        Args:
            rs_state (list): State in Robot Server format.

        Returns:
            numpy.array: State in environment format.

        """

        
        base_pose_keys = self.base_pose_list
        base_pose = []
        for p in base_pose_keys:
            base_pose.append(rs_state[p])
            
        base_pose = np.array(base_pose)

        # Compose environment state
        state = base_pose

        return state.astype(np.float32)
    
    def _get_observation_space(self) -> gym.spaces.Box:
        """Get environment observation space.

        Returns:
            gym.spaces: Gym observation space object.

        """
        max_pose = np.full(7, np.inf)
        min_pose = np.full(7, -np.inf)
        
        # Definition of environment observation_space
        max_obs = max_pose
        min_obs = min_pose
        base_obs_space = gym.spaces.Box(low=min_obs, high=max_obs, dtype=np.float64)
        if self.camera:
            camera_obs_shape = self.camera_config.observation_space
            return gym.spaces.Dict({
                'state': base_obs_space,
                'camera': gym.spaces.Box(
                    low=0, high=255,
                    shape=camera_obs_shape,
                    dtype=np.uint8
                )
            })
        else:
            return gym.spaces.Dict({
                'state': base_obs_space,
            })

    
    def _get_action_space(self) -> gym.spaces.Box:
        """Get environment action space.

        Returns:
            gym.spaces: Gym action space object.

        """        
        max_base_velocity = np.array([np.inf, np.inf])
        min_base_velocity = np.array([-np.inf, -np.inf])
        
        max_action = max_base_velocity
        min_action = min_base_velocity

        return gym.spaces.Box(low=min_action, high=max_action, dtype=np.float32)



class BunkerRRob(BunkerRBaseEnv):
    real_robot = True

# roslaunch interbotix_rover_robot_server interbotix_rover_real_robot_server.launch gui:=true reference_frame:=base
# action_cycle_rate:=20
