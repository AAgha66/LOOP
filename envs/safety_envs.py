# Code Reference: https://github.com/liuzuxin/safe-mbrl
import numpy as np
import re
import gym
import safety_gym
import realworldrl_suite.environments as rwrl

ROBOTS = ['Point','Car', 'Doggo']
TASKS = ['Goal', 'Button']

XYZ_SENSORS = dict(
    Point=['velocimeter'],
    Car=['velocimeter'],#,'accelerometer'],#,'ballquat_rear', 'right_wheel_vel', 'left_wheel_vel'],
    Doggo=['velocimeter','accelerometer']
    )

ANGLE_SENSORS = dict(
    Point=['gyro','magnetometer'],
    Car=['magnetometer','gyro'],
    Doggo=['magnetometer','gyro']
    )

CONSTRAINTS = dict(
    Goal=['vases', 'hazards'],
    Button=['hazards','gremlins','buttons'],)

DEFAULT_CONFIG = dict(
    action_repeat=5,
    max_episode_length=1000,
    use_dist_reward=False,
    stack_obs=False,
)

CONSTRAINT_INDICES = {"cartpole": 0, "walker": 1, "quadruped": 0}
SAFETY_COEFFS = {"cartpole": 0.3,"walker": 0.3,"quadruped": 0.5}

class Dict2Obj(object):
    #Turns a dictionary into a class
    def __init__(self, dictionary):
        for key in dictionary:
            setattr(self, key, dictionary[key])
        
    def __repr__(self):
        return "%s" % self.__dict__

class SafetyGymEnv():
    def __init__(self, robot='Point', task='Goal', level=1, seed=0, config=DEFAULT_CONFIG):
        self.robot = robot.capitalize()
        self.task = task.capitalize()
        assert self.robot in ROBOTS, "can not recognize the robot type {}".format(robot)
        assert self.task in TASKS, "can not recognize the task type {}".format(task)
        self.config = Dict2Obj(config)
        env_name = 'Safexp-'+self.robot+self.task+str(level)+'-v0'
        print("Creating environment: ", env_name)
        self.env = gym.make(env_name)
        self.env.seed(seed)
        # import ipdb;ipdb.set_trace()

        if not self.config.use_dist_reward:
            self.env.reward_distance = 0

        print("Environment configuration: ", self.config)
        self.init_sensor()
        
         #for uses with ppo in baseline
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.obs_flat_size,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (self.env.action_space.shape[0],), dtype=np.float32)
        self._max_episode_steps = config['max_episode_length']/config['action_repeat']

    def init_sensor(self):
        self.xyz_sensors = XYZ_SENSORS[self.robot]
        self.angle_sensors = ANGLE_SENSORS[self.robot]
        self.constraints_name = CONSTRAINTS[self.task]

        self.base_state_name = self.xyz_sensors + self.angle_sensors
        self.flatten_order = self.base_state_name + ["goal"] + self.constraints_name #+ self.distance_name

        # get state space vector size
        self.env.reset()
        obs = self.get_obs()
        self.obs_flat_size = sum([np.prod(i.shape) for i in obs.values()])
        self.state_dim = self.obs_flat_size
        if self.config.stack_obs:
            self.state_dim = self.state_dim*self.config.action_repeat
        self.key_to_slice = {}
        offset = 0
        for k in self.flatten_order:
            k_size = np.prod(obs[k].shape)
            self.key_to_slice[k] = slice(offset, offset + k_size)
            print("obs key: ", k, " slice: ", self.key_to_slice[k])
            offset += k_size

        self.base_state_dim = sum([np.prod(obs[k].shape) for k in self.base_state_name])
        self.action_dim = self.env.action_space.shape[0]
        self.key_to_slice["base_state"] = slice(0, self.base_state_dim)

    def reset(self):
        self.t = 0    # Reset internal timer
        self.env.reset()
        obs = self.get_obs_flatten()
        
        if self.config.stack_obs:
            for k in range(self.config.action_repeat):
                cat_obs = obs if k == 0 else np.concatenate((cat_obs, obs))
            return cat_obs
        else:
            return obs
    
    def step(self, action):
        # 2 dimensional numpy array, [vx, w]
        reward = 0
        cost = 0
        if self.config.stack_obs:
            cat_obs = np.zeros(self.config.action_repeat*self.obs_flat_size)

        for k in range(self.config.action_repeat):
            control = action
            state, reward_k, done, info = self.env.step(control)
            reward += reward_k
            cost += info["cost"]
            self.t += 1    # Increment internal timer
            observation = self.get_obs_flatten()
            if self.config.stack_obs:
                cat_obs[k*self.obs_flat_size :(k+1)*self.obs_flat_size] = observation 
            goal_met = ("goal_met" in info.keys()) # reach the goal
            done = done or self.t == self.config.max_episode_length
            if done or goal_met:
                if k != self.config.action_repeat-1 and self.config.stack_obs:
                    for j in range(k+1,self.config.action_repeat):
                        cat_obs[j*self.obs_flat_size :(j+1)*self.obs_flat_size] = observation 
                break
        info = {"cost":cost, "goal_met":goal_met}
        if self.config.stack_obs:
            return cat_obs, reward, done, info
        else:
            return observation, reward, done, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def recenter(self, pos):
        ''' Return the egocentric XY vector to a position from the robot '''
        return self.env.ego_xy(pos)

    def dist_xy(self, pos):
        ''' Return the distance from the robot to an XY position, 3 dim or 2 dim '''
        return self.env.dist_xy(pos)

    def get_cost(self, obs):
        N = obs.shape[0]
        hazards_pos_list = self.env.hazards_pos # list of shape (3,) ndarray
        ego_hazards_pos_list = [self.env.ego_xy(pos[:2]) for pos in hazards_pos_list] # list of shape (2,) ndarray
        
        hazards_key = self.key_to_slice['hazards']

        hazard_obs = obs[:,hazards_key].reshape(N,-1,2)
        hazards_dist = np.sqrt(np.sum(np.square(hazard_obs),axis=2)).reshape(N,-1)
        cost = ((hazards_dist<self.env.hazards_size)*(self.env.hazards_size-hazards_dist)).sum(1) * 10

        return cost

    def get_observation_cost(self, obs):
        N = obs.shape[0]
        hazards_pos_list = self.env.hazards_pos # list of shape (3,) ndarray
        ego_hazards_pos_list = [self.env.ego_xy(pos[:2]) for pos in hazards_pos_list] # list of shape (2,) ndarray
        
        hazards_key = self.key_to_slice['hazards']

        hazard_obs = obs[:,hazards_key].reshape(N,-1,2)
        hazards_dist = np.sqrt(np.sum(np.square(hazard_obs),axis=2)).reshape(N,-1)
        cost = ((hazards_dist<self.env.hazards_size)*(self.env.hazards_size-hazards_dist)).sum(1) * 10

        return cost

    def get_obs(self):
        '''
        We will ingnore the z-axis coordinates in every poses.
        The returned obs coordinates are all in the robot coordinates.
        '''

        obs = {}
        robot_pos = self.env.robot_pos
        goal_pos = self.env.goal_pos
        vases_pos_list = self.env.vases_pos # list of shape (3,) ndarray
        hazards_pos_list = self.env.hazards_pos # list of shape (3,) ndarray
        gremlins_pos_list = self.env.gremlins_obj_pos # list of shape (3,) ndarray
        buttons_pos_list = self.env.buttons_pos # list of shape (3,) ndarray


        ego_goal_pos = self.recenter(goal_pos[:2])
        ego_vases_pos_list = [self.env.ego_xy(pos[:2]) for pos in vases_pos_list] # list of shape (2,) ndarray
        ego_hazards_pos_list = [self.env.ego_xy(pos[:2]) for pos in hazards_pos_list] # list of shape (2,) ndarray
        ego_gremlins_pos_list = [self.env.ego_xy(pos[:2]) for pos in gremlins_pos_list] # list of shape (2,) ndarray
        ego_buttons_pos_list = [self.env.ego_xy(pos[:2]) for pos in buttons_pos_list] # list of shape (2,) ndarray


        # append obs to the dict
        for sensor in self.xyz_sensors:  # Explicitly listed sensors
            if sensor=='accelerometer':
                obs[sensor] = self.env.world.get_sensor(sensor)[:1] # only x axis matters
            elif sensor=='ballquat_rear':
                obs[sensor] = self.env.world.get_sensor(sensor)
            else:
                obs[sensor] = self.env.world.get_sensor(sensor)[:2] # only x,y axis matters

        for sensor in self.angle_sensors:
            if sensor == 'gyro':
                obs[sensor] = self.env.world.get_sensor(sensor)[2:] #[2:] # only z axis matters
                #pass # gyro does not help
            else:
                obs[sensor] = self.env.world.get_sensor(sensor)

        obs["vases"] = np.array(ego_vases_pos_list) # (vase_num, 2)
        obs["hazards"] = np.array(ego_hazards_pos_list) # (hazard_num, 2)
        obs["goal"] = ego_goal_pos # (2,)
        obs["gremlins"] = np.array(ego_gremlins_pos_list) # (vase_num, 2)
        obs["buttons"] = np.array(ego_buttons_pos_list) # (hazard_num, 2)

        return obs

    def get_obs_flatten(self):
        # get the flattened obs
        self.obs = self.get_obs()
        flat_obs = np.zeros(self.obs_flat_size)
        for k in self.flatten_order:
            idx = self.key_to_slice[k]
            flat_obs[idx] = self.obs[k].flat
        return flat_obs

    def get_dist_reward(self):
        '''
        @return reward: negative distance from robot to the goal
        '''
        return -self.env.dist_goal()

    @property
    def observation_size(self):
        return self.state_dim

    @property
    def action_size(self):
        return self.env.action_space.shape[0]

    @property
    def action_range(self):
        return float(self.env.action_space.low[0]), float(self.env.action_space.high[0])

    # Sample an action randomly from a uniform distribution over all valid actions
    def sample_random_action(self):
        return self.env.action_space.sample()

class ActionRepeatWrapper(gym.Wrapper):
    def __init__(self, env, repeat, binary_cost=False):
        super().__init__(env)
        if not type(repeat) is int or repeat < 1:
            raise ValueError("Repeat value must be an integer and greater than 0.")
        self.action_repeat = repeat
        self._max_episode_steps = 1000//repeat
        self.binary_cost = binary_cost

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        track_info = info.copy()
        track_reward = reward
        for i in range(self.action_repeat-1):
            if done or self.action_repeat==1:
                return observation, reward, done, info
            observation1, reward1, done1, info1 = self.env.step(action)
            track_info["cost"] += info1["cost"]
            track_reward += reward1

        if self.binary_cost:
            track_info["cost"] = 1 if track_info["cost"] > 0 else 0
        return observation1, track_reward, done1, track_info


class RWRLBridge(gym.Env):
    def __init__(self, env, constraint_idx):
        self._env = env
        self._constraint_idx = constraint_idx

    @property
    def action_space(self):
        spec = self._env.action_spec()
        return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

    @property
    def observation_space(self):        
        n_obs = 0
        dtype=None
        for k, v in self._env.observation_spec().items():
            if k == "constraints":
                continue
            n_obs += v.shape[0]
            dtype=v.dtype
        return gym.spaces.Box(-np.inf, np.inf, np.array([n_obs,]), dtype=dtype)
    
    def reset(self):
        time_step = self._env.reset()        
        obs, _ = self._get_obs(time_step)
        return obs

    def _get_obs(self, timestep):
        arrays = []
        for k, v in self._env.observation_spec().items():
            if k == "constraints":
                cost = 1.0 - timestep.observation["constraints"][self._constraint_idx]                
            else:
                array = timestep.observation[k]
                if v.shape == ():
                    array = np.array([array])
                arrays.append(array)
        obs = np.concatenate(arrays, -1)
        return obs, cost

    def get_observation_cost(self, obs):
        cost = np.array(np.abs(obs[:,0])>0.6)*1
        return cost

    def render(self, mode='human', **kwargs):
        if 'camera_id' not in kwargs.keys():
            kwargs['camera_id'] = 0
        return self._env.physics.render(**kwargs)

    def step(self, action):
        time_step = self._env.step(action)
        obs, cost = self._get_obs(time_step)
        reward = time_step.reward or 0
        done = time_step.last()
        return obs, reward, done, {"cost": cost}

class CartpoleRWRLBridge(RWRLBridge):
    def get_observation_cost(self, obs):
        return np.array(np.abs(obs[:,0])>0.6) * 1

class WalkerRWRLBridge(RWRLBridge):
    def get_observation_cost(self, obs):
        return (np.abs(self._env.physics.named.data.qvel[3:]) >= (0.3 * 65)).any() * 1

class QuadrupedRWRLBridge(RWRLBridge):
    def get_observation_cost(self, obs):
        from pdb import set_trace
        set_trace()        
        cost = ((np.abs(self._env.physics.named.data.qpos[self._env.task._hinge_names]))>(0.5 * 60 * np.pi / 180)).any() * 1
        return cost

def make_rwrl(domain_name, action_repeat=2, episode_length=1000, pixel_obs=False):
    domain, task = domain_name.rsplit('.', 1)
    env = rwrl.load(
            domain_name=domain,
            task_name=task,
            safety_spec=dict(
                enable=True, observations=True, safety_coeff=SAFETY_COEFFS[domain]
            ),
            environment_kwargs={'flat_observation': False}
        )        
    if domain.lower()=="cartpole":
        env = CartpoleRWRLBridge(env, CONSTRAINT_INDICES[domain])
    elif domain.lower()=="walker":
        env = WalkerRWRLBridge(env, CONSTRAINT_INDICES[domain])
    elif domain.lower()=="quadruped":
        env = QuadrupedRWRLBridge(env, CONSTRAINT_INDICES[domain])
    else:
        raise ValueError(f"This env:{domain} is not supported")
    env.reset()
    if action_repeat>1:
        ar_env = ActionRepeatWrapper(env, repeat=action_repeat, binary_cost=True)
        return ar_env
    else:
        return env