import numpy as np
import torch
import time
import uuid
import random
import os
from utils import *
from typing import List
from gym import spaces
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.side_channel.side_channel import SideChannel, IncomingMessage, OutgoingMessage
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from torchvision import transforms
from PIL import Image
from torchvision.utils import make_grid
import random
from nnmodels.dpt_depth import DPTDepth

DIM_GOAL = 3
DIM_ACTION = 2
BITS = 2

img_width, img_height, channels = 256, 256, 3
transforms_ = [
    transforms.Resize((img_height, img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
transform = transforms.Compose(transforms_)

visibility_constant = 1

# yolo = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
# gan = GeneratorFunieGAN()
# gan.load_state_dict(torch.load("funie_generator.pth"))
# if torch.cuda.is_available():
#     gan.cuda()
# gan.eval()

class PosChannel(SideChannel):
    def __init__(self) -> None:
        super().__init__(uuid.UUID("621f0a70-4f87-11ea-a6bf-784f4387d1f7"))

    def on_message_received(self, msg: IncomingMessage) -> None:
        """
        Note: We must implement this method of the SideChannel interface to
        receive messages from Unity
        """
        self.goal_depthfromwater = msg.read_float32_list()

    def goal_depthfromwater_info(self):
        return self.goal_depthfromwater

    def assign_testpos_visibility(self, data: List[float]) -> None:
        msg = OutgoingMessage()
        msg.write_float32_list(data)
        super().queue_message_to_send(msg)

class UnderwaterNavigation:
    def __init__(self, depth_prediction_model, adaptation, randomization, rank, history, start_goal_pos=None, training=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._validate_parameters(adaptation, randomization, start_goal_pos, training)
        self._initialize_depth_model(depth_prediction_model)
        self._initialize_parameters(adaptation, randomization, history, training, start_goal_pos)
        self._setup_unity_env(rank)
    
    def reset(self):
        self.total_episodes += 1
        self.step_count = 0

        # Adjust the visibility of the environment before resetting
        self._adjust_visibility()
        self.env.reset()

        # Retrieve the first observation
        obs_goal_depthfromwater = np.array(self.pos_info.goal_depthfromwater_info())
        self._eval_save(obs_goal_depthfromwater)

        # Stepping with zero action to get the first observation
        obs_img_ray, _, done, _ = self.env.step([0, 0])
        obs_predicted_depth = self.dpt.run(obs_img_ray[0] ** 0.45)

        # Get the minimum value of certain indices in the second channel of obs_img_ray
        indices = [1, 3, 5, 33, 35]
        values = [obs_img_ray[1][i] for i in indices]
        min_value = np.min(values)

        # Multiply the minimum value by 8 and 0.5 to get the final value for obs_ray
        obs_ray_value = min_value * 8 * 0.5
        obs_ray = np.array([obs_ray_value])

        # Retrive the second observation
        obs_goal_depthfromwater = np.array(self.pos_info.goal_depthfromwater_info())
        self._eval_save(obs_goal_depthfromwater)

        # Construct the observations of depth images, goal infos, and rays\
        self.prevPos = (obs_goal_depthfromwater[4], obs_goal_depthfromwater[3], obs_goal_depthfromwater[5])
        self.obs_predicted_depths = np.array([obs_predicted_depth.tolist()] * self.history)
        self.obs_goals = np.array([obs_goal_depthfromwater[:3].tolist()] * self.history)
        self.obs_rays = np.array([obs_ray.tolist()] * self.history)
        self.obs_actions = np.array([[0, 0]] * self.history)
        self.obs_visibility = np.reshape(self.visibility_para_Gaussian, [1, 1, 1])
        
        # self.firstDetect = True

        # # Process observation image with YOLO
        # color_img = self._yolo_process(obs_img_ray[0])

        # # Get the current position of the robot
        # x_pos = obs_goal_depthfromwater[4]
        # y_pos = obs_goal_depthfromwater[3]
        # z_pos = obs_goal_depthfromwater[5]
        # orientation = obs_goal_depthfromwater[6]

        # # Detect the bottle in the observation image
        # horizontal, vertical, hdeg, detected = self._detect_bottle(color_img)
        # if detected:
        #     self.obs_goals = np.array([[horizontal, vertical, hdeg]] * self.history)
        #     self.randomGoal = False
        #     self.firstDetect = False
        # else:
        #     self.randomGoal = True

        # # Get the position of the bottle (goal)
        # self._update_prev_goal(x_pos, y_pos, z_pos, orientation)

        # # Randomize the goal position
        # if self.randomGoal:
        #     self.prevGoal[0] += random.uniform(-3, 3)
        #     self.prevGoal[1] += random.uniform(-0.25, 0.25)
        #     self.prevGoal[2] += random.uniform(-3, 3)
        #     print(self.prevGoal)
        #     horizontal, vertical, hdeg = self._update_obs_goal(obs_goal_depthfromwater)
        #     self.obs_goals = np.array([[horizontal, vertical, hdeg]] * self.history)
        # print("Score: {} / {}".format(self.total_correct, self.total_steps))
        # print("Scorev2: {} / {}".format(self.reach_goal, self.total_episodes))

        return self.obs_predicted_depths, self.obs_goals, self.obs_rays, self.obs_actions

    def step(self, action):
        self.time_before = time.time()
        
        # action[0] controls its vertical speed, action[1] controls its rotation speed
        action_ver, action_rot = action
        action_rot *= self.twist_range
        
        # observations per frame
        obs_img_ray, _, done, _ = self.env.step([action_ver, action_rot])
        obs_predicted_depth = self.dpt.run(obs_img_ray[0] ** 0.45)
        obs_predicted_depth = np.reshape(obs_predicted_depth, (1, self.dpt.depth_image_height, self.dpt.depth_image_width))
        self.obs_predicted_depths = np.append(obs_predicted_depth, self.obs_predicted_depths[: (self.history - 1), :, :], axis=0)
        
        # compute obstacle distance
        obs_ray, obstacle_distance, obstacle_distance_vertical = self._get_obs(obs_img_ray)
        """
            compute reward
            obs_goal_depthfromwater[0]: horizontal distance
            obs_goal_depthfromwater[1]: vertical distance
            obs_goal_depthfromwater[2]: angle from robot's orientation to the goal (degree)
            obs_goal_depthfromwater[3]: robot's current y position
            obs_goal_depthfromwater[4]: robot's current x position            
            obs_goal_depthfromwater[5]: robot's current z position     
            obs_goal_depthfromwater[6]: robot's current orientation       
        """
        obs_goal_depthfromwater = self.pos_info.goal_depthfromwater_info()
        horizontal_distance = obs_goal_depthfromwater[0]
        vertical_distance = obs_goal_depthfromwater[1]
        vertical_distance_abs = np.abs(vertical_distance)
        angle_to_goal = obs_goal_depthfromwater[2]
        angle_to_goal_abs_rad = np.abs(np.deg2rad(angle_to_goal))
        y_pos = obs_goal_depthfromwater[3]
        x_pos = obs_goal_depthfromwater[4]
        z_pos = obs_goal_depthfromwater[5]
        orientation = obs_goal_depthfromwater[6]

        # 1. give a negative reward when robot is too close to nearby obstacles, seafloor or the water surface
        if obstacle_distance < 0.5:
            reward_obstacle = -10
            done = True
            print("Too close to the obstacle!")
            print("Horizontal distance to nearest obstacle:", obstacle_distance)
        elif np.abs(y_pos) < 0.24:
            reward_obstacle = -10
            done = True
            print("Too close to the seafloor!")
            print("Distance to water surface:", np.abs(y_pos))
        elif obstacle_distance_vertical < 0.12:
            reward_obstacle = -10
            done = True
            print("Too close to the vertical obstacle!")
            print("Vertical distance to nearest obstacle:", obstacle_distance_vertical)
        else:
            reward_obstacle = 0

        # 2. give a positive reward if the robot reaches the goal
        goal_distance_threshold = 0.6 if self.training else 0.8
        if horizontal_distance < goal_distance_threshold:
            reward_goal_reached = (10 - 8 * vertical_distance_abs - angle_to_goal_abs_rad)
            done = True
            print("Reached the goal area!")
            self.reach_goal += 1
        else:
            reward_goal_reached = 0

        # 3. give a positive reward if the robot is reaching the goal
        reward_goal_reaching_vertical = np.abs(action_ver) if vertical_distance * action_ver > 0 else -np.abs(action_ver)

        # 4. give negative rewards if the robot too often turns its direction or is near any obstacle
        reward_goal_reaching_horizontal = (-angle_to_goal_abs_rad + np.pi / 3) / 10
        if 0.5 <= obstacle_distance < 1.0:
            reward_goal_reaching_horizontal *= (obstacle_distance - 0.5) / 0.5
            reward_obstacle -= (1 - obstacle_distance) * 2
        reward = (reward_obstacle + reward_goal_reached + reward_goal_reaching_horizontal + reward_goal_reaching_vertical)
        self.step_count += 1
        if self.step_count > 500:
            done = True
            print("Exceeds the max num_step...")
            
        # detect the bottle
        # color_img = self._yolo_process(obs_img_ray[0])
        # horizontal, vertical, hdeg, detected = self._detect_bottle(color_img)
        # obs_goal = np.reshape(np.array(obs_goal_depthfromwater[0:3]), (1, DIM_GOAL))
        # if detected:
        #     obs_goal = np.reshape(np.array([horizontal, vertical, hdeg]), (1, DIM_GOAL))
        #     self.obs_goals = np.append(obs_goal, self.obs_goals[: (self.history - 1), :], axis=0)
        #     self.firstDetect = False
        # self._update_prev_goal(x_pos, y_pos, z_pos, orientation)
        # if not detected:
        #     horizontal, vertical, hdeg = self._update_obs_goal(obs_goal_depthfromwater)
        #     obs_goal = np.reshape(np.array([horizontal, vertical, hdeg]), (1, DIM_GOAL))
        #     self.obs_goals = np.append(obs_goal, self.obs_goals[: (self.history - 1), :], axis=0)
        #     print("object not detected. Angle is {}".format(hdeg))
        # self._update_history(action, obs_ray)
        # self._eval_save(obs_goal_depthfromwater)
        # self.time_after = time.time()
        # self.total_steps += 1
        
        # construct the observations of depth images, goal infos, and rays for consecutive 4 frames
        obs_predicted_depth = np.reshape(obs_predicted_depth, (1, self.dpt.depth_image_height, self.dpt.depth_image_width))
        self.obs_predicted_depths = np.append(obs_predicted_depth, self.obs_predicted_depths[: (self.history - 1), :, :], axis=0)

        obs_goal = np.reshape(np.array(obs_goal_depthfromwater[0:3]), (1, DIM_GOAL))
        self.obs_goals = np.append(obs_goal, self.obs_goals[:(self.history - 1), :], axis=0)

        obs_ray = np.reshape(np.array(obs_ray), (1, 1))  # single beam sonar and adaptation representation
        self.obs_rays = np.append(obs_ray, self.obs_rays[:(self.history - 1), :], axis=0)

        obs_action = np.reshape(action, (1, DIM_ACTION))
        self.obs_actions = np.append(obs_action, self.obs_actions[:(self.history - 1), :], axis=0)

        self.time_after = time.time()
        self._eval_save(obs_goal_depthfromwater)
        
        print(f'x: {x_pos}, y: {y_pos}, z: {z_pos}, orientation: {orientation}, horizontal distance: {horizontal_distance}, vertical distance: {vertical_distance}, angle to goal: {angle_to_goal}, reward: {reward}, done: {done}')

        return self.obs_predicted_depths, self.obs_goals, self.obs_rays, self.obs_actions, reward, done, 0

    def _get_obs(self, obs_img_ray):
        obs_ray = np.array([
            np.min([
                obs_img_ray[1][1],
                obs_img_ray[1][3],
                obs_img_ray[1][5],
                obs_img_ray[1][33],
                obs_img_ray[1][35]
            ]) * 8 * 0.5
        ])
        obstacle_distance = (
            np.min(
                [
                    obs_img_ray[1][1],
                    obs_img_ray[1][3],
                    obs_img_ray[1][5],
                    obs_img_ray[1][7],
                    obs_img_ray[1][9],
                    obs_img_ray[1][11],
                    obs_img_ray[1][13],
                    obs_img_ray[1][15],
                    obs_img_ray[1][17],
                ]
            ) * 8 * 0.5
        )
        obstacle_distance_vertical = (
            np.min(
                [
                    obs_img_ray[1][81],
                    obs_img_ray[1][79],
                    obs_img_ray[1][77],
                    obs_img_ray[1][75],
                    obs_img_ray[1][73],
                    obs_img_ray[1][71],
                ]
            ) * 8 * 0.5
        )
        return obs_ray,obstacle_distance,obstacle_distance_vertical

    def _validate_parameters(self, adaptation, randomization, start_goal_pos, training):
        if adaptation and not randomization:
            raise Exception("Adaptation should be used with domain randomization during training")
        if not training and start_goal_pos is None:
            raise AssertionError

    def _initialize_parameters(self, adaptation, randomization, history, training, start_goal_pos):
        self.adaptation = adaptation
        self.randomization = randomization
        self.history = history
        self.training = training
        self.start_goal_pos = start_goal_pos
        # Initialize additional class variables
        self.twist_range = 30  # degree
        self.vertical_range = 0.1
        self.action_space = spaces.Box(
            np.array([-self.twist_range, -self.vertical_range]).astype(np.float32),
            np.array([self.twist_range, self.vertical_range]).astype(np.float32),
        )
        # Initialize observation space variables
        self.observation_space_img_depth = (self.history, self.dpt.depth_image_height, self.dpt.depth_image_width)
        self.observation_space_goal = (self.history, DIM_GOAL)
        self.observation_space_ray = (self.history, 1)
        # Initialize performance tracking variables
        self.total_steps = 0
        self.total_correct = 0
        self.total_episodes = 0
        self.reach_goal = 0

    def _setup_unity_env(self, rank):
        config_channel = EngineConfigurationChannel()
        self.pos_info = PosChannel()
        unity_env = UnityEnvironment(
            os.path.abspath("./") + "/underwater_env/water",
            side_channels=[config_channel, self.pos_info],
            worker_id=rank,
            base_port=5004
        )
        if not self.training:
            visibility = 3 * (13 ** random.uniform(0, 1)) if self.randomization else 3 * (13 ** visibility_constant)
            self.pos_info.assign_testpos_visibility(self.start_goal_pos + [visibility])
        config_channel.set_configuration_parameters(time_scale=10, capture_frame_rate=100)
        self.env = UnityToGymWrapper(unity_env, allow_multiple_obs=True)

    def _initialize_depth_model(self, depth_prediction_model):
        model_path = os.path.abspath("./") + "/DPT/weights/"
        if depth_prediction_model == "dpt":
            model_file = "dpt_large-midas-2f21e586.pt"
            model_type = "dpt_large"
        elif depth_prediction_model == "midas":
            model_file = "midas_v21_small-70d6b9c8.pt"
            model_type = "midas_v21_small"
        self.dpt = DPTDepth(self.device, model_type=model_type, model_path=model_path + model_file)

    def _adjust_visibility(self):
        # Adjust the visibility of the environment
        if self.randomization:
            visibility_para = random.uniform(-1, 1)
            visibility = 3 * (13 ** ((visibility_para + 1) / 2))
            if self.adaptation:
                self.visibility_para_Gaussian = np.clip(np.random.normal(visibility_para, 0.02, 1), -1, 1)
            else:
                self.visibility_para_Gaussian = np.array([0])
        else:
            visibility = 3 * (13 ** visibility_constant)
            self.visibility_para_Gaussian = np.array([0])
        # Assign the visibility to the environment
        if self.training:
            self.pos_info.assign_testpos_visibility([0] * 9 + [visibility])
        else:
            self.pos_info.assign_testpos_visibility(self.start_goal_pos + [visibility])

    def _eval_save(self, obs_goal_depthfromwater):
        if not self.training:
            with open(os.path.join(assets_dir(), "learned_models/test_pos.txt"), "a") as f:
                f.write(f"{obs_goal_depthfromwater[4]} {obs_goal_depthfromwater[5]} {obs_goal_depthfromwater[6]}\n")
        
    # def _detect_bottle(self, color_img):
    #     detected = False
    #     for index, name in enumerate(color_img.pandas().xyxy[0]["name"].values):
    #         if name != "bottle":
    #             continue
    #         print(color_img.pandas().xyxy[0]["name"][index])
    #         # Get bounding box coordinates
    #         xmin = color_img.pandas().xyxy[0]["xmin"][index]
    #         xmax = color_img.pandas().xyxy[0]["xmax"][index]
    #         ymin = color_img.pandas().xyxy[0]["ymin"][index]
    #         ymax = color_img.pandas().xyxy[0]["ymax"][index]
    #         # Get the center of the bounding box
    #         xmid = int((xmin + xmax) / 4)
    #         ymid = int((ymin + ymax) / 4)
    #         # Get the depth of the center of the bounding box
    #         size = (xmax - xmin) * (ymax - ymin) / 4
    #         depth = 1 / size * 1200
    #         # Get the horizontal and vertical distance of the center of the bounding box
    #         vdeg = (64 - ymid) / 2
    #         horizontal = depth * abs(math.cos(math.radians(vdeg)))
    #         vertical = depth * math.sin(math.radians(vdeg))
    #         # Get the horizontal angle of the center of the bounding box
    #         hdeg = (80 - xmid) / 2
    #         detected = True
    #         if detected:
    #             self.total_correct += 1
    #             break
    #     return horizontal, vertical, hdeg, detected

    # def _extract_xy(self, x0, z0, ang):
    #     if ang > 270:
    #         ang = 360 - ang
    #         x = x0 - self.obs_goals[0][0] * math.sin(math.radians(ang))
    #         z = z0 + self.obs_goals[0][0] * math.cos(math.radians(ang))
    #     elif ang > 180:
    #         ang = ang - 180
    #         x = x0 - self.obs_goals[0][0] * math.sin(math.radians(ang))
    #         z = z0 - self.obs_goals[0][0] * math.cos(math.radians(ang))
    #     elif ang > 90:
    #         ang = 180 - ang
    #         x = x0 + self.obs_goals[0][0] * math.sin(math.radians(ang))
    #         z = z0 - self.obs_goals[0][0] * math.cos(math.radians(ang))
    #     else:
    #         x = x0 + self.obs_goals[0][0] * math.sin(math.radians(ang))
    #         z = z0 + self.obs_goals[0][0] * math.cos(math.radians(ang))
    #     return x, z, ang
    
    # def _yolo_process(self, img):
    #     color_img = 256 * img ** 0.45
    #     color_img = Image.fromarray(color_img.astype(np.uint8))
    #     color_img = transform(color_img).unsqueeze(0).to(self.device).float()
    #     color_img = gan(color_img).detach()
    #     grid = make_grid(color_img, normalize=True)
    #     transformed_grid = grid.mul(255).add_(0.5).clamp_(0, 255)
    #     rearranged_grid = transformed_grid.permute(1, 2, 0).to("cpu", torch.uint8)
    #     color_img = rearranged_grid.numpy()
    #     return yolo(color_img)

    # def _update_history(self, action, obs_ray):
    #     obs_ray = np.reshape(np.array(obs_ray), (1, 1))
    #     self.obs_rays = np.append(obs_ray, self.obs_rays[: (self.history - 1), :], axis=0)
    #     obs_action = np.reshape(action, (1, DIM_ACTION))
    #     self.obs_actions = np.append(obs_action, self.obs_actions[: (self.history - 1), :], axis=0)

    # def _update_obs_goal(self, obs_goal_depthfromwater):
    #     # Extract x, y, z coordinates from the underwater depth information
    #     x1 = obs_goal_depthfromwater[4]
    #     y1 = obs_goal_depthfromwater[3]
    #     z1 = obs_goal_depthfromwater[5]
    #     # Get the previous goal coordinates
    #     x = self.prevGoal[0]
    #     y = self.prevGoal[1]
    #     z = self.prevGoal[2]
    #     # Calculate the angle between the current and previous goals
    #     ang = normalize_angle(obs_goal_depthfromwater[6])
    #     goalDir = [x - x1, y - y1, z - z1]
    #     horizontal = math.sqrt(goalDir[0] ** 2 + goalDir[2] ** 2)
    #     vertical = goalDir[1]
    #     a = np.array([goalDir[0], goalDir[2]])
    #     a = a / np.linalg.norm(a)
    #     b = np.array([0, 1])
    #     goalAng = math.degrees(math.acos(np.dot(a, b)))
    #     if a[0] < 0:
    #         goalAng = 360 - goalAng
    #     hdeg = ang - goalAng
    #     if hdeg > 180:
    #         hdeg -= 360
    #     elif hdeg < -180:
    #         hdeg += 360
    #     # Return the horizontal and vertical distances and the angle between the goals
    #     return horizontal, vertical, hdeg
    
    # def _update_prev_goal(self, x_pos, y_pos, z_pos, orientation):
    #     goal_vertical = self.obs_goals[0][1]
    #     goal_hdeg = self.obs_goals[0][2]
    #     currAng = normalize_angle(orientation)
    #     ang = currAng - goal_hdeg
    #     ang = normalize_angle(ang)
    #     x, z, ang = self._extract_xy(x_pos, z_pos, ang)
    #     y = y_pos + goal_vertical
    #     self.prevGoal = [x, y, z]