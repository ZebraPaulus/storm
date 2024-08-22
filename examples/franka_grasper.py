#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.#
""" Example spawning a robot in gym 

"""
import copy
from isaacgym import gymapi
import time

# from isaacgym import gymutil

import torch

torch.multiprocessing.set_start_method("spawn", force=True)
torch.set_num_threads(8)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import matplotlib

matplotlib.use("tkagg")

# import matplotlib.pyplot as plt

# import time
import yaml
import argparse
import numpy as np

# from quaternion import quaternion, from_rotation_vector, from_rotation_matrix
import matplotlib.pyplot as plt

from quaternion import (
    from_euler_angles,
    as_float_array,
    # as_rotation_matrix,
    # from_float_array,
    # as_quat_array,
)

from storm_kit.gym.core import Gym, World
from storm_kit.gym.sim_robot import RobotSim
from storm_kit.util_file import (
    # get_configs_path,
    get_gym_configs_path,
    join_path,
    load_yaml,
    get_assets_path,
)

# from storm_kit.gym.helpers import load_struct_from_dict

# from storm_kit.util_file import get_mpc_configs_path as mpc_configs_path

from storm_kit.differentiable_robot_model.coordinate_transform import (
    quaternion_to_matrix,
    CoordinateTransform,
)
from storm_kit.mpc.task.reacher_task import ReacherTask

np.set_printoptions(precision=2)


def mpc_robot_interactive(args, gym_instance, debug=False):

    # ROS
    if debug == False:
        from storm_kit.robot_interface.joint_states import MPCRobotController
        from storm_kit.robot_interface.ball_tracker import BallTracker

        lab_controller = MPCRobotController()
        ball_tracker = BallTracker()

    vis_ee_target = True
    robot_file = "franka.yml"
    task_file = "franka_grasper.yml"
    world_file = "robo_lab.yml"

    gym = gym_instance.gym
    sim = gym_instance.sim
    world_yml = join_path(get_gym_configs_path(), world_file)
    with open(world_yml) as file:
        world_params = yaml.load(file, Loader=yaml.FullLoader)

    robot_yml = join_path(get_gym_configs_path(), args.robot + ".yml")
    with open(robot_yml) as file:
        robot_params = yaml.load(file, Loader=yaml.FullLoader)
    sim_params = robot_params["sim_params"]
    sim_params["asset_root"] = get_assets_path()
    if args.cuda:
        device = "cuda"
    else:
        device = "cpu"

    # set light to reduce shadows:
    intensity = 1
    light_index = gymapi.Vec3(1.0, 1.0, 1.0)
    ambient = gymapi.Vec3(0.4, 0.4, 0.4)
    direction = gymapi.Vec3(0.0, np.pi, np.pi / 2)

    gym.set_light_parameters(sim, intensity, light_index, ambient, direction)

    sim_params["collision_model"] = None
    # create robot simulation:
    robot_sim = RobotSim(
        gym_instance=gym, sim_instance=sim, **sim_params, device=device
    )

    # create gym environment:
    robot_pose = sim_params["robot_pose"]
    env_ptr = gym_instance.env_list[0]
    robot_ptr = robot_sim.spawn_robot(env_ptr, robot_pose, coll_id=2)

    device = torch.device("cuda", 0)

    tensor_args = {"device": device, "dtype": torch.float32}

    # spawn camera:
    robot_camera_pose = np.array([1.6, -1.5, 1.8, 0.707, 0.0, 0.0, 0.707])
    q = as_float_array(
        from_euler_angles(-0.5 * 90.0 * 0.01745, 50.0 * 0.01745, 90 * 0.01745)
    )
    robot_camera_pose[3:] = np.array([q[1], q[2], q[3], q[0]])

    robot_sim.spawn_camera(env_ptr, 60, 640, 480, robot_camera_pose)

    # get pose
    w_T_r = copy.deepcopy(robot_sim.spawn_robot_pose)

    w_T_robot = torch.eye(4)
    quat = torch.tensor([w_T_r.r.w, w_T_r.r.x, w_T_r.r.y, w_T_r.r.z]).unsqueeze(0)
    rot = quaternion_to_matrix(quat)
    w_T_robot[0, 3] = w_T_r.p.x
    w_T_robot[1, 3] = w_T_r.p.y
    w_T_robot[2, 3] = w_T_r.p.z
    w_T_robot[:3, :3] = rot[0]

    world_instance = World(gym, sim, env_ptr, world_params, w_T_r=w_T_r)

    # table_dims = np.ravel([1.5, 2.5, 0.7])
    # cube_pose = np.ravel([0.35, -0.0, -0.35, 0.0, 0.0, 0.0, 1.0])

    # cube_pose = np.ravel([0.9, 0.3, 0.4, 0.0, 0.0, 0.0, 1.0])

    # table_dims = np.ravel([0.35, 0.1, 0.8])

    # cube_pose = np.ravel([0.35, 0.3, 0.4, 0.0, 0.0, 0.0, 1.0])

    # table_dims = np.ravel([0.3, 0.1, 0.8])

    # get camera data:
    mpc_control = ReacherTask(task_file, robot_file, world_file, tensor_args)

    n_dof = mpc_control.controller.rollout_fn.dynamics_model.n_dofs

    # start_qdd = torch.zeros(n_dof, **tensor_args)

    # update goal:

    # exp_params = mpc_control.exp_params

    # current_state = copy.deepcopy(robot_sim.get_state(env_ptr, robot_ptr))
    # ee_list = []

    mpc_tensor_dtype = {"device": device, "dtype": torch.float32}

    # Franka initial state:

    ee_error = 10.0
    j = 0
    t_step = 0

    def goal_state(tensor_args):
        gs = torch.as_tensor(
            np.array(
                [-0.3, 0.3, 0.2, -2.0, 0.0, 2.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ),
            **tensor_args
        ).unsqueeze(0)

        def out(t):
            return gs

        return out

    mpc_control.update_params(t_step, goal_state=goal_state)

    # spawn object:
    asset_options = gymapi.AssetOptions()
    asset_options.armature = 0.001
    asset_options.fix_base_link = True
    asset_options.thickness = 0.002

    obj_asset_root = get_assets_path()

    g_p = np.ravel(mpc_control.controller.rollout_fn.goal_ee_pos.cpu().numpy())
    object_pose = gymapi.Transform()
    object_pose.p = gymapi.Vec3(g_p[0], g_p[1], g_p[2])
    object_pose.r = gymapi.Quat(0, 0, 0, 1)

    # ball in end effector:
    ee_handle = world_instance.spawn_object(
        "urdf/ball/ball.urdf",
        obj_asset_root,
        object_pose,
        name="ee_current_as_ball",
    )
    ee_body_handle = gym.get_actor_rigid_body_handle(env_ptr, ee_handle, 0)
    gym.set_rigid_body_color(
        env_ptr,
        ee_handle,
        0,
        gymapi.MESH_VISUAL_AND_COLLISION,
        gymapi.Vec3(0, 0.8, 0),
    )

    # moving ball:
    object_pose.r = gymapi.Quat(0, -0.7071068, 0, 0.7071068)

    target_object = world_instance.spawn_object(
        "urdf/ball/movable_ball.urdf",
        obj_asset_root,
        object_pose,
        name="ee_target_object",
    )

    tray_color = gymapi.Vec3(0.8, 0.1, 0.1)

    # obj_base_handle = gym.get_actor_rigid_body_handle(env_ptr, target_object, 0)
    # gym.set_rigid_body_color(
    #     env_ptr, target_object, 0, gymapi.MESH_VISUAL_AND_COLLISION, tray_color
    # )
    # the moving ball
    obj_body_handle = gym.get_actor_rigid_body_handle(env_ptr, target_object, 6)
    gym.set_rigid_body_color(
        env_ptr, target_object, 6, gymapi.MESH_VISUAL_AND_COLLISION, tray_color
    )

    # object_pose = w_T_r * object_pose
    # some weird rotation
    # if vis_ee_target:
    #     gym.set_rigid_transform(env_ptr, obj_base_handle, object_pose)
    # n_dof = mpc_control.controller.rollout_fn.dynamics_model.n_dofs
    # prev_acc = np.zeros(n_dof)
    ee_pose = gymapi.Transform()
    w_robot_coord = CoordinateTransform(
        trans=w_T_robot[0:3, 3].unsqueeze(0), rot=w_T_robot[0:3, 0:3].unsqueeze(0)
    )

    # rollout = mpc_control.controller.rollout_fn
    tensor_args = mpc_tensor_dtype
    sim_dt = mpc_control.exp_params["control_dt"]

    # log_traj = {"q": [], "q_des": [], "qdd_des": [], "qd_des": [], "qddd_des": []}

    q_des = None
    # qd_des = None
    t_step = gym_instance.get_sim_time()

    g_p = np.ravel(mpc_control.controller.rollout_fn.goal_ee_pos.cpu().numpy())
    g_q = np.ravel(mpc_control.controller.rollout_fn.goal_ee_quat.cpu().numpy())

    # region target control
    gym_instance

    pp = gymapi.PlaneParams()
    pp.normal = gymapi.Vec3(0, 1, 0)
    pp.distance = 0.8
    gym.add_ground(sim, pp)

    # currying function, that ignores second argument
    # use as placeholder to implement time dependent g_p and g_q
    def set_goal_ee(b: np.ndarray, c = None):
        if c is None:
            c = b*0
        def add_tensor_args(tensor_args=None):
            def out(t=1):
                return (
                    b
                    if tensor_args is None
                    else torch.as_tensor(b, **tensor_args).unsqueeze(0)
                )

            return out

        return add_tensor_args

    # init
    g_p = set_goal_ee(g_p)
    g_q = set_goal_ee(g_q)

    times = {"start": 0, "end": 0}
    times_file = open("times.csv", "w")
    times_file.write(
        "start,step,get_pose,update_params,get_command,get_error,set_gym,set_lines,end\n"
    )
    while t_step > -100:
        try:
            times["start"] = time.time()
            gym_instance.step()
            times["step"] = time.time()
            # updating robot position
            if debug == False:
                current_robot_state = lab_controller.get_current_joint_state()
                robot_sim.set_robot_state(
                    current_robot_state["position"],
                    current_robot_state["velocity"],
                    env_ptr,
                    robot_ptr,
                )
                current_ball_pos = ball_tracker.get_absolute_position()
                pose = gymapi.Transform()
                pose.p = gymapi.Vec3(
                    current_ball_pos[0], current_ball_pos[1], current_ball_pos[2]
                )
                pose.r = gymapi.Quat(0.7071068, -0.7071068,0,0)
                # pose.r.x = 0
                # pose.r.y = -0.7071068
                # pose.r.z = 0.7071068
                # pose.r.w = 0
                # sim.set_rigid_transform(env_ptr, obj_body_handle, p)
                # world_instance.set_pose(
                #     obj_body_handle,
                #     pose,
                # )
                # world_instance.set_pose(obj_body_handle, p)

            else:
                current_robot_state = copy.deepcopy(
                    robot_sim.get_state(env_ptr, robot_ptr)
                )
                pose = copy.deepcopy(world_instance.get_pose(obj_body_handle))
                pose = copy.deepcopy(w_T_r.inverse() * pose)
            # updating target pose for current ball position
            times["get_pose"] = time.time()

            if (
                np.linalg.norm(g_p()() - np.ravel([pose.p.x, pose.p.y, pose.p.z]))
                > 0.00001
            ) or (
                np.linalg.norm(
                    g_q()() - np.ravel([pose.r.w, pose.r.x, pose.r.y, pose.r.z])
                )
                > 0.00001
            ):

                g_p = set_goal_ee([pose.p.x, pose.p.y, pose.p.z])
                g_q = set_goal_ee([pose.r.w, pose.r.x, pose.r.y, pose.r.z])

                mpc_control.update_params(
                    t=t_step, dt=sim_dt, goal_ee_pos=g_p, goal_ee_quat=g_q
                )
            else:
                mpc_control.update_params(t=t_step, dt=sim_dt)

            t_step += sim_dt

            times["update_params"] = time.time()

            command = mpc_control.get_command(
                t_step, current_robot_state, control_dt=sim_dt, WAIT=True
            )
            times["get_command"] = time.time()

            curr_state = np.hstack(
                (
                    current_robot_state["position"],
                    current_robot_state["velocity"],
                    current_robot_state["acceleration"],
                )
            )

            curr_state_tensor = torch.as_tensor(curr_state, **tensor_args).unsqueeze(0)
            # get position command:
            q_des = copy.deepcopy(command["position"])
            qd_des = copy.deepcopy(command["velocity"])  # * 0.5
            # qdd_des = copy.deepcopy(command["acceleration"])

            ee_error = mpc_control.get_current_error(current_robot_state)
            times["get_error"] = time.time()

            pose_state = mpc_control.controller.rollout_fn.get_ee_pose(
                curr_state_tensor
            )
            # get current pose:
            e_pos = np.ravel(pose_state["ee_pos_seq"].cpu().numpy())
            e_quat = np.ravel(pose_state["ee_quat_seq"].cpu().numpy())
            ee_pose.p = copy.deepcopy(gymapi.Vec3(e_pos[0], e_pos[1], e_pos[2]))
            ee_pose.r = gymapi.Quat(e_quat[1], e_quat[2], e_quat[3], e_quat[0])

            ee_pose = copy.deepcopy(w_T_r) * copy.deepcopy(ee_pose)

            if vis_ee_target:
                gym.set_rigid_transform(env_ptr, ee_body_handle, copy.deepcopy(ee_pose))
            times["set_gym"] = time.time()
            print(
                # "\r",  # overwriting the line
                "[{:.5f}, {:.5f}, {:.5f}]".format(pose.p.x, pose.p.y, pose.p.z),
                end="\n",
            )

            gym_instance.clear_lines()
            top_trajs = mpc_control.top_trajs.cpu().float()  # .numpy()
            n_p, n_t = top_trajs.shape[0], top_trajs.shape[1]
            w_pts = w_robot_coord.transform_point(top_trajs.view(n_p * n_t, 3)).view(
                n_p, n_t, 3
            )

            top_trajs = w_pts.cpu().numpy()
            color = np.array([0.0, 1.0, 0.0])
            for k in range(top_trajs.shape[0]):
                pts = top_trajs[k, :, :]
                color[0] = float(k) / float(top_trajs.shape[0])
                color[1] = 1.0 - float(k) / float(top_trajs.shape[0])
                gym_instance.draw_lines(pts, color=color)
            times["set_lines"] = time.time()

            robot_sim.command_robot_position(q_des, env_ptr, robot_ptr)

            if debug == False:
                lab_controller.send_joint_command(q_des, qd_des)
            times["end"] = time.time()
            times_file.write(
                "{},{},{},{},{},{},{},{},{}\n".format(
                    times["start"],
                    times["step"],
                    times["get_pose"],
                    times["update_params"],
                    times["get_command"],
                    times["get_error"],
                    times["set_gym"],
                    times["set_lines"],
                    times["end"],
                )
            )

        except KeyboardInterrupt:
            print("Closing")
            # done = True
            break
    times_file.close()
    print("File closed")
    mpc_control.close()
    if debug == False:
        lab_controller.close()
    return 1


if __name__ == "__main__":

    # instantiate empty gym:
    parser = argparse.ArgumentParser(description="pass args")
    parser.add_argument("--robot", type=str, default="franka", help="Robot to spawn")
    parser.add_argument("--cuda", action="store_true", default=True, help="use cuda")
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="headless gym",
        dest="headless",
    )
    parser.add_argument(
        "--control_space", type=str, default="acc", help="Robot to spawn"
    )
    parser.add_argument(
        "--debug", action="store_true", default=False, help="headless gym"
    )
    args = parser.parse_args()

    sim_params = load_yaml(join_path(get_gym_configs_path(), "physx.yml"))
    sim_params["headless"] = args.headless
    gym_instance = Gym(**sim_params)

    mpc_robot_interactive(args, gym_instance, debug=args.debug)
