import gym
from gym import spaces
from typing import Optional

from storm_kit.gym.core import Gym
import argparse
from storm_kit.util_file import get_configs_path, get_gym_configs_path, join_path, load_yaml, get_assets_path

import copy
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymapi

import torch
torch.multiprocessing.set_start_method('spawn',force=True)
torch.set_num_threads(8)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import matplotlib
matplotlib.use('tkagg')

import matplotlib.pyplot as plt
from utils import animate

import time
import yaml
import argparse
import numpy as np
from quaternion import quaternion, from_rotation_vector, from_rotation_matrix
import matplotlib.pyplot as plt

from quaternion import from_euler_angles, as_float_array, as_rotation_matrix, from_float_array, as_quat_array

from storm_kit.gym.core import Gym, World
from storm_kit.gym.sim_robot import RobotSim
from storm_kit.util_file import get_configs_path, get_gym_configs_path, join_path, load_yaml, get_assets_path
from storm_kit.gym.helpers import load_struct_from_dict

from storm_kit.util_file import get_mpc_configs_path as mpc_configs_path

from storm_kit.differentiable_robot_model.coordinate_transform import quaternion_to_matrix, CoordinateTransform
from storm_kit.mpc.task.reacher_task import ReacherTask
np.set_printoptions(precision=2)

from typing import Union
from cprint import *

BIN_X_COORD = [-0.1, 0.15, 0.35, 0.6]
BIN_Y_COORD = [1, 1.3, 1.45, 1.6, 1.74, 1.95, 2.08]
BIN_Z_COORD = [-0.55]

def get_random_bin_coord():
    return BIN_X_COORD[np.random.randint(0,len(BIN_X_COORD))], BIN_Y_COORD[np.random.randint(0,len(BIN_Y_COORD))], BIN_Z_COORD[np.random.randint(0,len(BIN_Z_COORD))]

class IsaacGymEnv(gym.Env):
    # TODO modify metadata
    metadata = {"render_modes": ["human", "rgb_array", "single_rgb_array"], "render_fps": 4}
    
    def __init__(self, render_mode: Optional[str] = None):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode  # Define the attribute render_mode in your environment

        # self.observation_space = spaces.Dict(
        #     {
        #         "agent": spaces.Box(low=0, high=np.pi, shape=(6,), dtype=np.float32),
        #         "target": spaces.Box(low=0, high=2.5, shape=(3,), dtype=np.float32)
        #     }
        # )
        # Observation space is the target pose
        self.observation_space = spaces.Box(low=-2.5, high=2.5, shape=(3,), dtype=np.float32)
        # Action space is the goal pose
        self.action_space = spaces.Box(low=-2.5, high=2.5, shape=(3,), dtype=np.float32)

        # STORM objects
        # instantiate empty gym:
        parser = argparse.ArgumentParser(description='pass args')
        parser.add_argument('--robot', type=str, default='ur16e', help='Robot to spawn')
        parser.add_argument('--cuda', action='store_true', default=True, help='use cuda')
        parser.add_argument('--headless', action='store_true', default=False, help='headless gym')
        parser.add_argument('--control_space', type=str, default='acc', help='Robot to spawn')
        parser.add_argument('--camera_pose', nargs='+', type=float, default=[2.0, 0.0, 0.0, 0.707,0.0,0.0,-0.707], help='Where to spawn camera')
        parser.add_argument('--num_env', type=int, default='1', help='Number of environments')
        args = parser.parse_args()
        
        sim_params = load_yaml(join_path(get_gym_configs_path(),'physx.yml'))
        #sim_params['up_axis'] = gymapi.UP_AXIS_Z
        gym_instance = Gym(**sim_params)
        
        real_robot = False
        self.robot_file = args.robot + '.yml'
        self.task_file = args.robot + '_reacher_collision.yml'
        self.world_file = 'collision_primitives_3d.yml'

        self.gym_instance = gym_instance
        self.gym = gym_instance.gym
        self.sim = gym_instance.sim
        
        self.world_yml = join_path(get_gym_configs_path(), self.world_file)
        with open(self.world_yml) as file:
            self.world_params = yaml.load(file, Loader=yaml.FullLoader)

        self.robot_yml = join_path(get_gym_configs_path(),args.robot + '.yml')
        with open(self.robot_yml) as file:
            self.robot_params = yaml.load(file, Loader=yaml.FullLoader)
        self.sim_params = self.robot_params['sim_params']
        self.sim_params['asset_root'] = get_assets_path()
        if(args.cuda):
            device = 'cuda'
        else:
            device = 'cpu'
        self.sim_params['collision_model'] = None

        self.robot_sim = []
        self.robot_pose = self.sim_params['robot_pose']
        self.env_ptr = []
        self.robot_ptr = []
        self.world_instance = []
        self.mpc_control = []
        self.w_T_r = []
        self.w_T_robot = []
        self.obj_body_handle = []
        self.sensor_idx = []
        self.target_object = []
        self.target_pose = []
        self.tensor_args = []

        spacing = 3
        for i in range(args.num_env):
            self.tensor_args.append({'device':torch.device('cuda', i) , 'dtype':torch.float32})
            self.gym_instance._create_env(spacing)
            # create robot simulation:
            self.robot_sim.append(RobotSim(gym_instance=self.gym, sim_instance=self.sim, **self.sim_params, device=device))

            # create gym environment:
            self.env_ptr.append(gym_instance.env_list[i])
            self.robot_ptr.append(self.robot_sim[i].spawn_robot(self.env_ptr[i], self.robot_pose, coll_id=2))
    
            # get pose
            self.w_T_r.append(copy.deepcopy(self.robot_sim[i].spawn_robot_pose))

            self.w_T_robot.append(torch.eye(4))
            quat = torch.tensor([self.w_T_r[i].r.w,self.w_T_r[i].r.x,self.w_T_r[i].r.y,self.w_T_r[i].r.z]).unsqueeze(0)
            rot = quaternion_to_matrix(quat)
            self.w_T_robot[i][0,3] = self.w_T_r[i].p.x
            self.w_T_robot[i][1,3] = self.w_T_r[i].p.y
            self.w_T_robot[i][2,3] = self.w_T_r[i].p.z
            self.w_T_robot[i][:3,:3] = rot[0]

             # spawn camera:
            self.robot_sim[i].spawn_camera(self.env_ptr[i], 60, 640, 480, np.array(args.camera_pose)) 

            # World Instance
            self.world_instance.append(World(self.gym, self.sim, self.env_ptr[i], self.world_params, w_T_r=self.w_T_r[i])   )

            # Control Parameters
            self.mpc_control.append(ReacherTask(self.task_file, self.robot_file, self.world_file, self.tensor_args[i]))
            self.sim_dt = self.mpc_control[i].exp_params['control_dt']
            self.mpc_control[i].update_params(goal_state=np.array([-0.85, 0.6, 0.2, -1.8, 0.0, 2.4, 0.0,
                                        10.0, 10.0, 10.0,  0.0, 0.0, 0.0, 0.0]))

            if len(self.world_instance[i].sphere_handles) > 0:
                sphere_handle = self.world_instance[i].sphere_handles[0] 
                self.obj_body_handle.append(self.gym_instance.gym.get_actor_rigid_body_handle(self.env_ptr[i], sphere_handle, 0))

            # Spawn object
            x,y,z = get_random_bin_coord()
            tray_color = gymapi.Vec3(1, 0, 0)
            asset_options = gymapi.AssetOptions()
            asset_options.armature = 0.001
            asset_options.fix_base_link = False
            asset_options.thickness = 0.002


            object_pose = gymapi.Transform()
            object_pose.p = gymapi.Vec3(x, y, z)
            object_pose.r = gymapi.Quat(1, 0, 0, 0)
            
            target_object, sensor_idx = self.world_instance[i].spawn_box(object_pose, color=tray_color, name='ee_target_object')
            self.target_object.append(target_object)
            self.sensor_idx.append(sensor_idx)
            self.gym.set_rigid_body_color(self.env_ptr[i], self.target_object[i], 0, gymapi.MESH_VISUAL_AND_COLLISION, tray_color)

            # Append target pose
            self.target_pose.append(self.get_pose(self.env_ptr[i], self.target_object[i]))

        # Simulation time step
        self.t_step = 0

        # Save root tensor
        self.gym_instance.get_tensor()

    def _get_obs(self):
        NotImplementedError

    def _get_info(self):
        NotImplementedError



class TahomaEnv(IsaacGymEnv):
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__(render_mode)
        self.ee_pose = []
        self.distance_to_goal = []
        for i in range(len(self.env_ptr)):
            self.ee_pose.append(gymapi.Transform())
            self.distance_to_goal.append(None)
        self.min_distance_to_goal = 0.02
        self.obs = None
        self.primitives = Primitives()

        ## Test
        self.test = gymapi.Transform()
        self.test.p = gymapi.Vec3(0.5, 1.520999, -0.4)#(0.376706, 1.520999, -0.135020)
        self.test.r = gymapi.Quat(-0.484209, 0.874026, 0.035971, 0.018075)#(0, 0, 0, 1)#(-0.484209, 0.874026, 0.035971, 0.018075)
        self.goal = self.test
        self.test_primitives = False
        self.flag = "up"

    def step(self, action):
        self.gym_instance.step()
        self.gym_instance.clear_lines()
        self.t_step += self.sim_dt
        reward = []
        ob = []
        info = []
        done = False # np.array([False, False])
        for i in range(len(self.env_ptr)):
            self.gym_instance.draw_sphere(self.target_pose[i], 0.05, 12, (0, 1, 1))
            pose_reached = self.pose_reached(i)
            if (self.test_primitives):
                # print("p", self.goal.p, "r", self.goal.r)
                if (pose_reached): 
                    cprint.info('######################REACHED#####################')
                    self.test, self.flag = self.primitives.square_pattern(self.test, self.flag)
                    # self.goal = self.primitives.get_cartesian_move(self.test, 0.05, 0.00, 0.0)
                    # self.goal = self.primitives.rotate(self.test, 0.0, 0.0, 0.02)
                self.set_goal(i, self.goal)
            else:
                if (pose_reached): cprint.info('######################REACHED#####################')
                self.set_goal(i, action)
            q_des, qd_des, qdd_des = self.move_robot(i)

            # Draw MPC path
            # self.draw_lines(i)
            # Draw Collision spheres
            # self.draw_collision(i)
            
            reward.append(self.get_reward(i, pose_reached))
            ob.append(self._get_obs(i))
            data = self.get_force(i)
            if np.linalg.norm(np.ravel([data.force.x, data.force.z])) > 0.001 and self.distance_to_goal[i] < 0.1: 
                print(self.t_step, 'GOAL TOUCHED', data.force)
                done = True
            info.append(self._get_info(i))
        return ob, reward, done, info

    def reset(self, seed=None, options=None):
        self.randomize()
        self.gym_instance.set_tensor()
        obs = []
        for i in range(len(self.env_ptr)):
            obs.append(self._get_obs(i))
        return obs

    def close(self):
        for i in range(len(self.env_ptr)):
            self.mpc_control[i].close()

    # def get_goal(self):
    #     pose = copy.deepcopy(self.gym.world_instance.get_pose(self.obj_body_handle)) # Review
    #     pose = copy.deepcopy(self.w_T_r.inverse() * pose) # Review

    def set_goal(self, env_num:int, pose:Union[np.ndarray,gymapi.Transform]):
        if type(pose) is np.ndarray:
            goal_pose = gymapi.Transform()
            goal_pose.p = gymapi.Vec3(pose[0], pose[1], pose[2]) 
            goal_pose.r = gymapi.Quat(pose[3], pose[4], pose[5], pose[6])
            # goal_pose.r = gymapi.Quat(0,0.707,0, 0.707)
        elif type(pose) is torch.Tensor:
            goal_pose = gymapi.Transform()
            goal_pose.p = gymapi.Vec3(pose[0], pose[1], pose[2]) 
            # goal_pose.r = gymapi.Quat(pose[3], pose[4], pose[5], pose[6])
            goal_pose.r = gymapi.Quat(0,0.707,0, 0.707)
        else:
            goal_pose = copy.deepcopy(pose)

        self.gym_instance.draw_sphere(goal_pose, 0.05, 12, (1, 0, 1))
        pose = copy.deepcopy(self.w_T_r[env_num].inverse() * goal_pose)
        g_pos = np.zeros(3)
        g_q = np.zeros(4)
        g_pos[0] = pose.p.x
        g_pos[1] = pose.p.y
        g_pos[2] = pose.p.z
        g_q[0] = pose.r.w
        g_q[1] = pose.r.x
        g_q[2] = pose.r.y
        g_q[3] = pose.r.z

        self.mpc_control[env_num].update_params(goal_ee_pos=g_pos, goal_ee_quat=g_q)

    def get_distance_to_goal(self, env_num: int):
        g_pos = np.ravel(self.mpc_control[env_num].controller.rollout_fn.goal_ee_pos.cpu().numpy())
        g_q = np.ravel(self.mpc_control[env_num].controller.rollout_fn.goal_ee_quat.cpu().numpy())

        current_robot_state = copy.deepcopy(self.robot_sim[env_num].get_state(self.env_ptr[env_num], self.robot_ptr[env_num])) 
        filtered_state_mpc = current_robot_state
        curr_state = np.hstack((filtered_state_mpc['position'], filtered_state_mpc['velocity'], filtered_state_mpc['acceleration']))
        curr_state_tensor = torch.as_tensor(curr_state, **self.tensor_args[env_num]).unsqueeze(0)
        pose_state = self.mpc_control[env_num].controller.rollout_fn.get_ee_pose(curr_state_tensor)
        ee_error = self.mpc_control[env_num].get_current_error(filtered_state_mpc) # This refreshes link trans and rot sequence
        e_pos = np.ravel(pose_state['ee_pos_seq'].cpu().numpy())
        e_quat = np.ravel(pose_state['ee_quat_seq'].cpu().numpy())
        self.ee_pose[env_num].p = copy.deepcopy(gymapi.Vec3(e_pos[0], e_pos[1], e_pos[2]))
        self.ee_pose[env_num].r = gymapi.Quat(e_quat[1], e_quat[2], e_quat[3], e_quat[0])

        self.distance_to_goal[env_num] = np.linalg.norm(g_pos - np.ravel([self.ee_pose[env_num].p.x, self.ee_pose[env_num].p.y, self.ee_pose[env_num].p.z]))

    def get_distance_to_target_object(self, env_num: int)-> float:
        pose = self.w_T_r[env_num] * self.ee_pose[env_num]
        return np.linalg.norm(np.ravel([self.target_pose[env_num].p.x, self.target_pose[env_num].p.y, self.target_pose[env_num].p.z]) - np.ravel([pose.p.x, pose.p.y, pose.p.z]))
        
    # Output: Returns if goal pose was reached
    # TODO Include Quaternion 
    def pose_reached(self, env_num:int)->bool:
        self.get_distance_to_goal(env_num)
        return self.distance_to_goal[env_num] < self.min_distance_to_goal

    # TODO Implement wether object fell or not to penalize
    def get_reward(self, num_env:int, pose_reached)->float:
        rew = -self.get_distance_to_target_object(num_env) 
        return rew

    def _get_obs(self, env_num:int)-> Union[gymapi.Transform, dict]:
        #return pose, current_robot_state # TODO Add multiple object poses
        # return {
        #         "agent": current_robot_state['position'],
        #         "target": np.array([self.target_pose.p.x, self.target_pose.p.y, self.target_pose.p.z]) 
        #         }
        return np.array([self.target_pose[env_num].p.x, self.target_pose[env_num].p.y, self.target_pose[env_num].p.z]) 

    def _get_info(self, env_num:int)->dict:
        return {"distance to target": self.get_distance_to_target_object(env_num),
                "distance to goal": self.distance_to_goal[env_num]}

    def move_robot(self, env_num:int) -> Union[np.array, np.array, np.array]:
        current_robot_state = copy.deepcopy(self.robot_sim[env_num].get_state(self.env_ptr[env_num], self.robot_ptr[env_num]))
        command = self.mpc_control[env_num].get_command(self.t_step, current_robot_state, control_dt=self.sim_dt, WAIT=False)
        q_des = copy.deepcopy(command['position'])
        qd_des = copy.deepcopy(command['velocity'])
        qdd_des = copy.deepcopy(command['acceleration'])
        self.robot_sim[env_num].command_robot_position(q_des, self.env_ptr[env_num], self.robot_ptr[env_num])
        return q_des, qd_des, qdd_des

    def draw_lines(self, env_num:int):
        w_robot_coord = CoordinateTransform(trans=self.w_T_robot[env_num][0:3,3].unsqueeze(0),
                                        rot=self.w_T_robot[env_num][0:3,0:3].unsqueeze(0))
        top_trajs = self.mpc_control[env_num].top_trajs.cpu().float()
        n_p, n_t = top_trajs.shape[0], top_trajs.shape[1]
        w_pts = w_robot_coord.transform_point(top_trajs.view(n_p * n_t, 3)).view(n_p, n_t, 3)

        top_trajs = w_pts.cpu().numpy()
        color = np.array([0.0, 1.0, 0.0])
        for k in range(top_trajs.shape[0]):
            pts = top_trajs[k,:,:]
            color[0] = float(k) / float(top_trajs.shape[0])
            color[1] = 1.0 - float(k) / float(top_trajs.shape[0])
            self.gym_instance.draw_lines(pts, color=color)

    def get_force(self, env_num:int)->gymapi.ForceSensor:
        return self.gym.get_actor_force_sensor(self.env_ptr[env_num], self.target_object[env_num], 0).get_forces()

    def get_pose(self, env_ptr, actor_handle):
        pose = self.gym.get_actor_rigid_body_states(env_ptr, actor_handle, gymapi.STATE_POS)['pose'][0]
        goal_pose = gymapi.Transform()
        goal_pose.p = gymapi.Vec3(pose[0][0], pose[0][1], pose[0][2]) 
        goal_pose.r = gymapi.Quat(pose[1][0], pose[1][1], pose[1][2], pose[1][3])
        return goal_pose

    def detect_collision(self):
        pass

    def randomize(self):
        # Index of target in isaacgym tensor was found through trial and error
        env_tensor_size = self.gym_instance.saved_root_tensor.size()[0] / len(self.env_ptr)
        for i in range(len(self.env_ptr)):
            x, y, z = get_random_bin_coord()
            index = int(env_tensor_size * (i + 1) - 1)
            self.gym_instance.saved_root_tensor[index, 0] = x 
            self.gym_instance.saved_root_tensor[index, 1] = y
            self.target_pose[i] = self.get_pose(self.env_ptr[i], self.target_object[i])

    def draw_collision(self, i):
        # device = torch.device('cuda', 0) 
        # tensor_args = {'device':device, 'dtype':torch.float32}
        # current_robot_state = copy.deepcopy(self.robot_sim[i].get_state(self.env_ptr[i], self.robot_ptr[i]))
        # filtered_state_mpc = current_robot_state #mpc_control.current_state
        # curr_state = np.hstack((filtered_state_mpc['position'], filtered_state_mpc['velocity'], filtered_state_mpc['acceleration']))

        # curr_state_tensor = torch.as_tensor(curr_state, **tensor_args).unsqueeze(0)
        # pose_state = self.mpc_control[i].controller.rollout_fn.get_ee_pose(curr_state_tensor)
        # Collision spheres
        link_pos_seq = copy.deepcopy(self.mpc_control[i].controller.rollout_fn.link_pos_seq)
        link_rot_seq = copy.deepcopy(self.mpc_control[i].controller.rollout_fn.link_rot_seq)
        # print('pos', link_pos_seq)
        # print('rot', link_rot_seq)
        batch_size = link_pos_seq.shape[0]
        horizon = link_pos_seq.shape[1]
        n_links = link_pos_seq.shape[2]
        link_pos = link_pos_seq.view(batch_size * horizon, n_links, 3)
        link_rot = link_rot_seq.view(batch_size * horizon, n_links, 3, 3)
        # print(link_pos)
        self.mpc_control[i].controller.rollout_fn.robot_self_collision_cost.coll.update_batch_robot_collision_objs(link_pos, link_rot)

        spheres = self.mpc_control[i].controller.rollout_fn.get_spheres()
        
        arr = None
        for sphere in spheres:
            if arr is None:
                arr = np.array(sphere[1:,:,:4].cpu().numpy().squeeze())
            else:
                arr = np.vstack((arr,sphere[1:,:,:4].cpu().numpy().squeeze()))

        [self.gym_instance.draw_collision_spheres(sphere,self.w_T_r[i]) for sphere in arr]

class Primitives():
    def __init__(self):
        pass

    def square_pattern(self, pose: gymapi.Transform, action: str):
        if action == "up": 
            return self.move_up(pose), "right"
        elif action == "right":
            return self.move_right(pose), "down"
        elif action == "down":
            return self.move_down(pose), "left"
        elif action == "left":
            return self.move_left(pose), "up"
        else:
            return pose, "done"

    def get_cartesian_move(self, pose: gymapi.Transform, x: float, y: float, z: float)->gymapi.Transform:
        pose.p.x = pose.p.x + x
        pose.p.y = pose.p.y + y
        pose.p.z = pose.p.z + z

        return pose

    def rotate(self, pose: gymapi.Transform, x: float, y: float, z: float)->gymapi.Transform:
        euler = pose.r.to_euler_zyx()
        x += euler[0]
        y += euler[1]
        z += euler[2]
        pose.r = pose.r.from_euler_zyx(x, y, z)
        return pose

    def move_right(self, pose: gymapi.Transform, distance=0.1)->gymapi.Transform:
        return self.get_cartesian_move(pose, distance, 0, 0)
    
    def move_left(self, pose: gymapi.Transform, distance=-0.1)->gymapi.Transform:
        return self.get_cartesian_move(pose, distance, 0, 0)

    def move_up(self, pose: gymapi.Transform, distance=0.1)->gymapi.Transform:
        return self.get_cartesian_move(pose, 0, distance, 0)

    def move_down(self, pose: gymapi.Transform, distance=-0.1)->gymapi.Transform:
        return self.get_cartesian_move(pose, 0, distance, 0)

    def move_forward(self, pose: gymapi.Transform, distance=-0.1)->gymapi.Transform:
        return self.get_cartesian_move(pose, 0, 0, distance)

    def move_back(self, pose: gymapi.Transform, distance=0.1)->gymapi.Transform:
        return self.get_cartesian_move(pose, 0, 0, distance)

    def push(self, pose: gymapi.Transform, distance=-0.2)->gymapi.Transform:
        return self.move_forward(pose, distance)

    def shake(self, pose: gymapi.Transform, distance=-0.1)->gymapi.Transform:
        pass

    def move_in():
        pass

    def lift():
        pass

    def move_out():
        pass

    def move_to_drop():
        pass

