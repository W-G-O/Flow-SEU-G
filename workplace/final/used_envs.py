#Fixedtime-ICV方法使用环境
class CavSingleEnv(Env):

    def __init__(self, env_params, sim_params, network, simulator='traci'):

        super().__init__(env_params, sim_params, network, simulator)
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))
    @property
    def action_space(self):
        num_actions = self.initial_vehicles.num_rl_vehicles
        accel_ub = self.env_params.additional_params["max_accel"]
        accel_lb = - abs(self.env_params.additional_params["max_decel"])

        return Box(low=accel_lb,
                   high=accel_ub,
                   shape=(num_actions,))

    @property
    def observation_space(self):

        return Box(
            low=-100,
            high=2000,
            shape=(6*self.initial_vehicles.num_rl_vehicles, ),
            dtype=np.float32)

    def get_state(self):

        veh_pos=[]
        veh_v=[]
        # the get_ids() method is used to get the names of all rl-vehicles in the network
        ids = self.k.vehicle.get_rl_ids()
        max_speed = self.k.network.max_speed()
        max_dist = 3000
        for id in ids:
            veh_pos = []
            veh_v = []
            # get the distance of each rl-veh with the leader and follower
            pos = self.k.vehicle.get_position(id)
            pos /= max_dist
            vel = self.k.vehicle.get_speed(id)
            vel /= max_speed
            veh_pos.append(pos)
            veh_v.append(vel)

            # ids of f and l
            follower = self.k.vehicle.get_follower(id)
            leader = self.k.vehicle.get_leader(id)

            f_veh_pos = self.k.vehicle.get_position(follower)
            if f_veh_pos == -1001:
                f_veh_pos = pos - 50
            f_veh_pos /= max_dist
            l_veh_pos = self.k.vehicle.get_position(leader)
            if l_veh_pos == -1001:
                l_veh_pos = pos + 50
            l_veh_pos /= max_dist
            veh_pos.append(f_veh_pos)
            veh_pos.append(l_veh_pos)

            f_veh_speed = self.k.vehicle.get_speed(follower)
            l_veh_speed = self.k.vehicle.get_speed(leader)
            if f_veh_speed == -1001:
                f_veh_speed = max_speed
            f_veh_speed /= max_speed
            if l_veh_speed == -1001:
                l_veh_speed = max_speed
            l_veh_speed /= max_speed
            veh_v.append(f_veh_speed)
            veh_v.append(l_veh_speed)

        state = np.array(np.concatenate([veh_pos+veh_v]))
        return state

    def _apply_rl_actions(self, rl_actions):

        rl_ids = self.k.vehicle.get_rl_ids()

        # use the base environment method to convert actions into accelerations for the rl vehicles
        self.k.vehicle.apply_acceleration(rl_ids, rl_actions)

    def compute_reward(self, rl_actions, **kwargs):

        ids = self.k.vehicle.get_rl_ids()
        max_speed = 30
        num_rl_veh = self.k.vehicle.num_rl_vehicles

        # the reward to forward the vehicles
        average_rl_speed = (sum(self.k.vehicle.get_speed(ids)) + .001)/ (num_rl_veh + .001)
        rv = average_rl_speed / max_speed
        if average_rl_speed - max_speed >= 5:
           rv -=  10
        if average_rl_speed == 0:
           rv -=  5
        return rv

#RLlight-HV方法使用环境
class MultiTrafficLightGridSEUEnv(MultiEnv):

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)

        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))
        self.grid_array = network.net_params.additional_params["grid_array"]
        self.rows = self.grid_array["row_num"]
        self.cols = self.grid_array["col_num"]
        self.num_traffic_lights = self.rows * self.cols
        self.tl_type = env_params.additional_params.get('tl_type')
        super().__init__(env_params, sim_params, network, simulator)
        # Saving env variables for plotting
        self.steps = env_params.horizon
        # number of vehicles nearest each intersection that is observed in the
        # state space; defaults to 1
        self.num_observed = env_params.additional_params.get("num_observed", 1)
        # used during visualization
        self.observed_ids = []

        if self.tl_type != "actuated":
            for i in range(self.rows * self.cols):
                self.k.traffic_light.set_state(
                    node_id='center' + str(i), state="GGGrrrrrrrrrGGGrrrrrrrrr")

        self.discrete = env_params.additional_params.get("discrete", True)

        # number of nearest lights to observe, defaults to 4
        self.num_local_lights = env_params.additional_params.get(
            "num_local_lights", 4)

        # number of nearest edges to observe, defaults to 4
        self.num_local_edges = env_params.additional_params.get(
            "num_local_edges", 4)

    @property
    def action_space(self):

        if self.discrete:
            return Discrete(4)
        else:
            return Box(
                low=0,
                high=3.999,
                shape=(1,),
                dtype=np.float32)

    @property
    def observation_space(self):

        tl_box = Box(
            low=0.,
            high=1,
            shape=(4 * (1 + self.num_local_lights)
                   + 2 * 3 * 4 * self.num_observed,),
            dtype=np.float32)
        return tl_box

    def get_state(self):

        max_speed = max(
            self.k.network.speed_limit(edge)
            for edge in self.k.network.get_edge_list())
        max_accel = max(2.6, 7.5)
        grid_array = self.net_params.additional_params["grid_array"]
        max_dist = max(grid_array["short_length"], grid_array["long_length"],
                       grid_array["inner_length"])

        speeds = []
        dist_to_intersec = []
        acc = []
        NQI_mean = []
        NQI_mean_0=[]
        all_observed_ids = []
        for nodes, edges in self.network.node_mapping:
            local_speeds = []
            local_dists_to_intersec = []
            local_acc = []
            for edge in edges:
                observed_ids = \
                    self.get_closest_to_intersection_lane(edge, self.num_observed)
                all_observed_ids.append(observed_ids)

                # check which edges we have so we can always pad in the right
                # positions
                for observed_id in observed_ids:
                    if observed_id != "":
                        local_speeds.extend(
                            [self.k.vehicle.get_speed(observed_id) / max_speed ])
                        local_dists_to_intersec.extend([(self.k.network.edge_length(
                            self.k.vehicle.get_edge(observed_id)) -
                            self.k.vehicle.get_position(observed_id)) / max_dist ])
                        local_acc.extend([self.k.vehicle.get_accel(observed_id) ])
                    elif observed_id=="":
                        local_speeds.extend([0])
                        local_dists_to_intersec.extend([0])
                        local_acc.extend([0])

            speeds.append(local_speeds)
            dist_to_intersec.append(local_dists_to_intersec)
            acc.append(local_acc)

            local_NQI_mean=[]
            NQI = []
            for edge in edges:
                straight_right = 0
                left = 0
                vehs = self.k.vehicle.get_ids_by_edge(edge)
                for veh in vehs:
                    if self.k.vehicle.get_speed(veh) == 0.0:
                        if self.k.vehicle.get_lane(veh) == 0 or self.k.vehicle.get_lane(veh) == 1:
                            straight_right += 1
                        elif self.k.vehicle.get_lane(veh) == 2:
                            left += 1
                NQI.append(straight_right / 0.19 / self.k.network.edge_length(edge) / 2)
                NQI.append(left / 0.19 / self.k.network.edge_length(edge))
            local_NQI_mean.extend([((NQI[0] + NQI[4]) / 2)])
            local_NQI_mean.extend([((NQI[1] + NQI[5]) / 2)])
            local_NQI_mean.extend([((NQI[2] + NQI[6]) / 2)])
            local_NQI_mean.extend([((NQI[3] + NQI[7]) / 2)])
            NQI_mean_0.append(local_NQI_mean)
        NQI_mean_0.append([0.0,0.0,0.0,0.0])

        self.observed_ids = all_observed_ids

        for rl_id in self.k.traffic_light.get_ids():
            rl_id_num = int(rl_id.split("center")[ID_IDX])
            local_id_nums = [rl_id_num, self._get_relative_node(rl_id, "top"),
                             self._get_relative_node(rl_id, "bottom"),
                             self._get_relative_node(rl_id, "left"),
                             self._get_relative_node(rl_id, "right")]

            NQI_mean_1 = []
            for local_id_num in local_id_nums:
                NQI_mean_1 += NQI_mean_0[local_id_num]
            NQI_mean.append(NQI_mean_1)

        obs = {}
        for rl_id in self.k.traffic_light.get_ids():
            rl_id_num = int(rl_id.split("center")[ID_IDX])
            observation = np.array(np.concatenate(
                [NQI_mean[rl_id_num], speeds[rl_id_num], dist_to_intersec[rl_id_num]]))

            obs.update({rl_id: observation})

        return obs

    def _apply_rl_actions(self, rl_actions):

        for rl_id, rl_action in rl_actions.items():
            i = int(rl_id.split("center")[ID_IDX])
            if self.discrete:
                action = int(rl_action)
            else:
                # convert values less than 0.0 to zero and above to 1. 0's
                # indicate that we should not switch the direction
                action = rl_action > 0.0

            if action == 0:
                self.k.traffic_light.set_state(
                    node_id='center{}'.format(i),
                    state="GGGrrrrrrrrrGGGrrrrrrrrr")
            elif action == 1:
                self.k.traffic_light.set_state(
                    node_id='center{}'.format(i),
                    state="rrrGGGrrrrrrrrrGGGrrrrrr")
            elif action == 2:
                self.k.traffic_light.set_state(
                    node_id='center{}'.format(i),
                    state="rrrrrrGGGrrrrrrrrrGGGrrr")
            elif action == 3:
                self.k.traffic_light.set_state(
                    node_id='center{}'.format(i),
                    state="rrrrrrrrrGGGrrrrrrrrrGGG")

    def compute_reward(self, rl_actions, **kwargs):

        if rl_actions is None:
            return {}

        if self.env_params.evaluate:
            NIN = 0
            for nodes, edges in self.network.node_mapping:
                Nin = 0
                for edge in edges:
                    vehs = self.k.vehicle.get_ids_by_edge(edge)
                    Nin += len(vehs)
                NIN += Nin

            NOUT = 0
            for nodes, edges in self.network.node_mapping_leave:
                Nout = 0
                for edge in edges:
                    vehs = self.k.vehicle.get_ids_by_edge(edge)
                    Nout += len(vehs)
                NOUT += Nout

            grid_array = self.net_params.additional_params["grid_array"]
            max_dist = max(grid_array["short_length"], grid_array["long_length"],
                           grid_array["inner_length"])
            rew_tl = -(NIN - NOUT) / 3 / max_dist / 0.15 + rewards.penalize_standstill(self, gain=1)
        else:
            NIN = 0
            for nodes, edges in self.network.node_mapping:
                Nin = 0
                for edge in edges:
                    vehs = self.k.vehicle.get_ids_by_edge(edge)
                    Nin += len(vehs)
                NIN += Nin

            NOUT = 0
            for nodes, edges in self.network.node_mapping_leave:
                Nout = 0
                for edge in edges:
                    vehs = self.k.vehicle.get_ids_by_edge(edge)
                    Nout += len(vehs)
                NOUT += Nout

            grid_array = self.net_params.additional_params["grid_array"]
            InDist = grid_array["short_length"]
            OutDist = grid_array["long_length"]
            veh_ids = self.k.vehicle.get_ids()
            vel = np.array(self.k.vehicle.get_speed(veh_ids))
            penalty = len(vel[vel == 0]) / len(vel)
            rew_tl = -(NIN / 3 / InDist / 0.15 - NOUT / 3 / OutDist / 0.15) - penalty

        # each agent receives reward normalized by number of lights
        rew_tl /= self.num_traffic_lights

        rews = {}
        for rl_id in rl_actions.keys():
            rews[rl_id] = rew_tl
        return rews

#In-PPO方法使用环境
class CustomSEUPressEnv(MultiEnv):

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)

        for p in ADDITIONAL_ENV_PARAMS_CAVTL.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        self.mapping_inc, _0, _1, _2 = network.get_edge_mappings
        self.mapping_inc=dict(list(self.mapping_inc.items())[:12])
        self.num_traffic_lights = len(self.mapping_inc.keys())
        # used during visualization
        self.observed_ids = []
        # exp setting
        self.num_observed = env_params.additional_params.get("num_observed", 1)
        # number of nearest lights to observe, defaults to 4
        self.num_local_lights = env_params.additional_params.get(
            "num_local_lights", 4)
        # number of nearest edges to observe, defaults to 4
        self.num_local_edges = env_params.additional_params.get(
            "num_local_edges", 4)
        self.grid_array = network.net_params.additional_params["grid_array"]
        self.rows = self.grid_array["row_num"]
        self.cols = self.grid_array["col_num"]
        self.tl_type = env_params.additional_params.get('tl_type')
        self.lastphase = [[0] for i in range(self.num_traffic_lights)]
        self.lastphase.append([100])
        self.nowsecond = [0]

        if self.tl_type != "actuated":
            for i in range(self.rows * self.cols):
                self.k.traffic_light.set_state(
                    node_id='center' + str(i), state="GGGrrrrrrrrrGGGrrrrrrrrr")

    @property
    def action_space_av(self):
        return Box(low=-abs(self.env_params.additional_params["max_decel"]),
                   high=self.env_params.additional_params["max_accel"], shape=(1,))

    @property
    def action_space_tl(self):
        # the action space for each traffic light
        return Discrete(4)

    @property
    def observation_space_av(self):
        # the observation space for each cav
        return Box(low=-100, high=2000, shape=(6,),dtype=np.float32)

    @property
    def observation_space_tl(self):
        return Box(low=0.,
                   high=1,
                   shape=(4 * (1 + self.num_local_lights)
                          + 2 * 3 * 4 * self.num_observed,),
                   dtype=np.float32)

    @property
    def action_space(self):
        return self.action_space_av, self.action_space_tl

    @property
    def observation_space(self):
        return self.observation_space_av, self.observation_space_tl

    def get_state(self):
        obs = {}

        #CAV/CAV/CAV/CAV/CAV/CAV/CAV/CAV/CAV/CAV/CAV/CAV/CAV/CAV/CAV/CAV/CAV
        ids = self.k.vehicle.get_rl_ids()
        max_speed = self.k.network.max_speed()
        max_dist = 3000
        for id in ids:
            veh_pos = []
            veh_v = []
            # get the position of each rl-veh
            pos = self.k.vehicle.get_position(id)
            pos /= max_dist
            vel = self.k.vehicle.get_speed(id)
            vel /= max_speed
            veh_pos.append(pos)
            veh_v.append(vel)

            # ids of f and l
            follower = self.k.vehicle.get_follower(id)
            leader = self.k.vehicle.get_leader(id)

            f_veh_pos = self.k.vehicle.get_position(follower)
            if f_veh_pos == -1001:
                f_veh_pos = pos - 50
            f_veh_pos /= max_dist
            l_veh_pos = self.k.vehicle.get_position(leader)
            if l_veh_pos == -1001:
                l_veh_pos = pos + 50
            l_veh_pos /= max_dist
            veh_pos.append(f_veh_pos)
            veh_pos.append(l_veh_pos)

            f_veh_speed = self.k.vehicle.get_speed(follower)
            l_veh_speed = self.k.vehicle.get_speed(leader)
            if f_veh_speed == -1001:
                f_veh_speed = max_speed
            f_veh_speed /= max_speed
            if l_veh_speed == -1001:
                l_veh_speed = max_speed
            l_veh_speed /= max_speed
            veh_v.append(f_veh_speed)
            veh_v.append(l_veh_speed)

            state_cav = np.array(np.concatenate((veh_pos, veh_v)))
            obs.update({id: state_cav})

        # trafficlight/trafficlight/trafficlight/trafficlight/trafficlight
        speeds = []
        dist_to_intersec = []
        acc = []
        NQI_mean = []
        NQI_mean_0 = []
        all_observed_ids = []
        for nodes, edges in self.network.node_mapping:
            local_speeds = []
            local_dists_to_intersec = []
            local_acc = []
            for edge in edges:
                observed_ids = \
                    self.get_closest_to_intersection_lane(edge, self.num_observed)
                all_observed_ids.append(observed_ids)

                for observed_id in observed_ids:
                    if observed_id != "":
                        local_speeds.extend(
                            [self.k.vehicle.get_speed(observed_id) / max_speed])
                        local_dists_to_intersec.extend([abs(self.k.network.edge_length(
                            self.k.vehicle.get_edge(observed_id)) -
                            self.k.vehicle.get_position(observed_id)) / max_dist])
                        local_acc.extend([self.k.vehicle.get_accel(observed_id)])
                    elif observed_id == "":
                        local_speeds.extend([0])
                        local_dists_to_intersec.extend([0])
                        local_acc.extend([0])

            speeds.append(local_speeds)
            dist_to_intersec.append(local_dists_to_intersec)
            acc.append(local_acc)

            local_NQI_mean = []
            NQI = []
            for edge in edges:
                straight_right = 0
                left = 0
                vehs = self.k.vehicle.get_ids_by_edge(edge)
                for veh in vehs:
                    if self.k.vehicle.get_speed(veh) == 0.0:
                        if self.k.vehicle.get_lane(veh) == 0 or self.k.vehicle.get_lane(veh) == 1:
                            straight_right += 1
                        elif self.k.vehicle.get_lane(veh) == 2:
                            left += 1
                NQI.append(straight_right / 0.19 / self.k.network.edge_length(edge) / 2)
                NQI.append(left / 0.19 / self.k.network.edge_length(edge))
            local_NQI_mean.extend([((NQI[0] + NQI[4]) / 2)])
            local_NQI_mean.extend([((NQI[1] + NQI[5]) / 2)])
            local_NQI_mean.extend([((NQI[2] + NQI[6]) / 2)])
            local_NQI_mean.extend([((NQI[3] + NQI[7]) / 2)])
            NQI_mean_0.append(local_NQI_mean)
        NQI_mean_0.append([0.0, 0.0, 0.0, 0.0])
        self.observed_ids = all_observed_ids

        for rl_id in self.k.traffic_light.get_ids():
            rl_id_num = int(rl_id.split("center")[ID_IDX])
            local_id_nums = [rl_id_num, self._get_relative_node(rl_id, "top"),
                             self._get_relative_node(rl_id, "bottom"),
                             self._get_relative_node(rl_id, "left"),
                             self._get_relative_node(rl_id, "right")]

            NQI_mean_1 = []
            for local_id_num in local_id_nums:
                NQI_mean_1 += NQI_mean_0[local_id_num]
            NQI_mean.append(NQI_mean_1)

        for rl_id in self.k.traffic_light.get_ids():
            rl_id_num = int(rl_id.split("center")[ID_IDX])
            state_tl = np.array(np.concatenate(
                [NQI_mean[rl_id_num], speeds[rl_id_num], dist_to_intersec[rl_id_num]]))

            obs.update({rl_id: state_tl})

        return obs

    def _apply_rl_actions(self, rl_actions):
        for rl_id, rl_action in rl_actions.items():
            # light
            if rl_id in self.mapping_inc.keys():
                i = int(rl_id.split("center")[ID_IDX])

                action = int(rl_action)
                if action == 0:
                    self.k.traffic_light.set_state(
                        node_id='center{}'.format(i),
                        state="GGGrrrrrrrrrGGGrrrrrrrrr")
                elif action == 1:
                    self.k.traffic_light.set_state(
                        node_id='center{}'.format(i),
                        state="rrrGGGrrrrrrrrrGGGrrrrrr")
                elif action == 2:
                    self.k.traffic_light.set_state(
                        node_id='center{}'.format(i),
                        state="rrrrrrGGGrrrrrrrrrGGGrrr")
                elif action == 3:
                    self.k.traffic_light.set_state(
                        node_id='center{}'.format(i),
                        state="rrrrrrrrrGGGrrrrrrrrrGGG")
            else:  # cav
                self.k.vehicle.apply_acceleration(rl_id, rl_action)
        self.nowsecond.append((self.nowsecond[-1] + 1))

    def compute_reward(self, rl_actions, **kwargs):
        if rl_actions is None:
            return {}

        rews = {}
        # reward for traffic light
        NIN = 0
        for nodes, edges in self.network.node_mapping:
            Nin = 0
            for edge in edges:
                vehs = self.k.vehicle.get_ids_by_edge(edge)
                Nin += len(vehs)
            NIN += Nin

        NOUT = 0
        for nodes, edges in self.network.node_mapping_leave:
            Nout = 0
            for edge in edges:
                vehs = self.k.vehicle.get_ids_by_edge(edge)
                Nout += len(vehs)
            NOUT += Nout


        grid_array = self.net_params.additional_params["grid_array"]
        InDist = grid_array["short_length"]
        OutDist = grid_array["long_length"]
        veh_ids = self.k.vehicle.get_ids()
        vel = np.array(self.k.vehicle.get_speed(veh_ids))
        penalty = len(vel[vel == 0]) / len(vel)
        rew_tl = -(NIN / 3 / InDist / 0.15 - NOUT / 3 / OutDist / 0.15)  - penalty

        rew_tl /= self.num_traffic_lights

        for rl_id in rl_actions.keys():
            rews[rl_id] = rew_tl

        # reward for cav
        ids = self.k.vehicle.get_rl_ids()
        max_speed = self.k.network.max_speed()
        num_rl_veh = self.k.vehicle.num_rl_vehicles

        average_rl_speed = (sum(self.k.vehicle.get_speed(ids)) + .001) / (num_rl_veh + .001)

        rew_cav = average_rl_speed / max_speed
        if average_rl_speed - max_speed >= 5:
            rew_cav -=  10
        if average_rl_speed == 0:
           rew_cav -= 5

        for rl_id in ids:
            rews[rl_id] = rew_cav

        return rews

#Co-PPO方法使用环境
class CustomSEUPress_cav_light_Env(CustomSEUPressEnv):

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)

        for p in ADDITIONAL_ENV_PARAMS_CAVTL.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))
        self.phases = [0]
        self.num_cav = env_params.additional_params.get("num_cav", 7)
        self.acc = [[0] for i in range(self.num_cav)]
    @property
    def observation_space_av(self):

        return Box(low=-100, high=2000, shape=(7,), dtype=np.float32)

    @property
    def observation_space_tl(self):

        return Box(low=-1.5,
                   high=1.5,
                   shape=(4 * (1 + self.num_local_lights)
                          + 2 * 3 * 4 * self.num_observed+4 * self.num_cav,),
                   dtype=np.float32)

    @property
    def observation_space(self):

        return self.observation_space_av, self.observation_space_tl

    def get_state(self):

        obs = {}
        # CAV/CAV/CAV/CAV/CAV/CAV/CAV/CAV/CAV/CAV/CAV/CAV/CAV/CAV/CAV/CAV
        ids = self.k.vehicle.get_rl_ids()
        max_speed = self.k.network.max_speed()
        max_dist = 3000

        for id in ids:
            veh_pos = []
            veh_v = []
            now_phase = []
            # get the position of each rl-veh
            pos = self.k.vehicle.get_position(id)
            pos /= max_dist
            vel = self.k.vehicle.get_speed(id)
            vel /= max_speed
            veh_pos.append(pos)
            veh_v.append(vel)

            # ids of f and l
            follower = self.k.vehicle.get_follower(id)
            leader = self.k.vehicle.get_leader(id)

            f_veh_pos = self.k.vehicle.get_position(follower)
            if f_veh_pos == -1001:
                f_veh_pos = pos - 50
            f_veh_pos /= max_dist
            l_veh_pos = self.k.vehicle.get_position(leader)
            if l_veh_pos == -1001:
                l_veh_pos = pos + 50
            l_veh_pos /= max_dist
            veh_pos.append(f_veh_pos)
            veh_pos.append(l_veh_pos)

            f_veh_speed = self.k.vehicle.get_speed(follower)
            l_veh_speed = self.k.vehicle.get_speed(leader)
            if f_veh_speed == -1001:
                f_veh_speed = max_speed
            f_veh_speed /= max_speed
            if l_veh_speed == -1001:
                l_veh_speed = max_speed
            l_veh_speed /= max_speed
            veh_v.append(f_veh_speed)
            veh_v.append(l_veh_speed)

            #得到每个CAV前进方向最近的信号灯的相位
            to_light=self.get_nearest_light(id)
            now_p=self.lastphase[to_light][-1]
            now_phase.append(now_p)
            state_cav = np.array(np.concatenate((veh_pos, veh_v, now_phase)))
            obs.update({id: state_cav})

        # trafficlight/trafficlight/trafficlight/trafficlight/trafficlight/
        speeds = []
        dist_to_intersec = []
        cav_acc = []
        NQI_mean = []
        NQI_mean_0 = []
        all_observed_ids = []

        for nodes, edges in self.network.node_mapping:
            local_speeds = []
            local_dists_to_intersec = []
            local_cav_acc = []
            cav_ids = list(self.k.vehicle.get_rl_ids())
            nodeset=self.network.nodes
            nodes_x=0
            nodes_y=0
            for node in nodeset:
                if node["id"]==nodes:
                    nodes_x=float(node["x"])
                    nodes_y=float(node["y"])
                    break
            for cav_id in cav_ids:
                local_speeds.extend(
                    [self.k.vehicle.get_speed(cav_id) / max_speed])
                local_dists_to_intersec.extend([abs(self.k.vehicle.get_2d_position(cav_id)[0]-nodes_x)/3000,
                                                abs(self.k.vehicle.get_2d_position(cav_id)[1]-nodes_y)/2500])
                cav_id_num = int(cav_id.split("cav_")[ID_IDX])
                local_cav_acc.extend([float(self.acc[cav_id_num][-1])/10])
            if len(cav_ids) < self.num_cav:
                local_speeds.extend([0]*(self.num_cav-len(cav_ids)))
                local_dists_to_intersec.extend([0]*2*(self.num_cav-len(cav_ids)))
                local_cav_acc.extend([0]*(self.num_cav-len(cav_ids)))

            for edge in edges:
                observed_ids = \
                    self.get_closest_to_intersection_lane(edge, self.num_observed)
                all_observed_ids.append(observed_ids)

                for observed_id in observed_ids:
                    if observed_id != "":
                        local_speeds.extend(
                            [self.k.vehicle.get_speed(observed_id) / max_speed])
                        local_dists_to_intersec.extend([abs(self.k.network.edge_length(
                            self.k.vehicle.get_edge(observed_id)) -
                            self.k.vehicle.get_position(observed_id)) / max_dist])
                    elif observed_id == "":
                        local_speeds.extend([0])
                        local_dists_to_intersec.extend([0])
            speeds.append(local_speeds)
            dist_to_intersec.append(local_dists_to_intersec)
            cav_acc.append(local_cav_acc)
            local_NQI_mean = []
            NQI = []
            for edge in edges:
                straight_right = 0
                left = 0
                vehs = self.k.vehicle.get_ids_by_edge(edge)
                for veh in vehs:
                    if self.k.vehicle.get_speed(veh) == 0.0:
                        if self.k.vehicle.get_lane(veh) == 0 or self.k.vehicle.get_lane(veh) == 1:
                            straight_right += 1
                        elif self.k.vehicle.get_lane(veh) == 2:
                            left += 1
                NQI.append(straight_right / 0.19 / self.k.network.edge_length(edge) / 2)
                NQI.append(left / 0.19 / self.k.network.edge_length(edge))
            local_NQI_mean.extend([((NQI[0] + NQI[4]) / 2)])
            local_NQI_mean.extend([((NQI[1] + NQI[5]) / 2)])
            local_NQI_mean.extend([((NQI[2] + NQI[6]) / 2)])
            local_NQI_mean.extend([((NQI[3] + NQI[7]) / 2)])
            NQI_mean_0.append(local_NQI_mean)
        NQI_mean_0.append([0.0, 0.0, 0.0, 0.0])
        self.observed_ids = all_observed_ids

        for rl_id in self.k.traffic_light.get_ids():
            rl_id_num = int(rl_id.split("center")[ID_IDX])
            local_id_nums = [rl_id_num, self._get_relative_node(rl_id, "top"),
                                 self._get_relative_node(rl_id, "bottom"),
                                 self._get_relative_node(rl_id, "left"),
                                 self._get_relative_node(rl_id, "right")]

            NQI_mean_1 = []
            for local_id_num in local_id_nums:
                NQI_mean_1 += NQI_mean_0[local_id_num]
            NQI_mean.append(NQI_mean_1)

        for rl_id in self.k.traffic_light.get_ids():
            rl_id_num = int(rl_id.split("center")[ID_IDX])
            state_tl = np.array(np.concatenate(
                 [NQI_mean[rl_id_num], speeds[rl_id_num], dist_to_intersec[rl_id_num],cav_acc[rl_id_num]]))
            obs.update({rl_id: state_tl})

        return obs

    def _apply_rl_actions(self, rl_actions):
        for rl_id, rl_action in rl_actions.items():
            # light
            if rl_id in self.mapping_inc.keys():
                i = int(rl_id.split("center")[ID_IDX])

                action = int(rl_action)
                if action == 0:
                    self.k.traffic_light.set_state(
                        node_id='center{}'.format(i),
                        state="GGGrrrrrrrrrGGGrrrrrrrrr")
                elif action == 1:
                    self.k.traffic_light.set_state(
                        node_id='center{}'.format(i),
                        state="rrrGGGrrrrrrrrrGGGrrrrrr")
                elif action == 2:
                    self.k.traffic_light.set_state(
                        node_id='center{}'.format(i),
                        state="rrrrrrGGGrrrrrrrrrGGGrrr")
                elif action == 3:
                    self.k.traffic_light.set_state(
                        node_id='center{}'.format(i),
                        state="rrrrrrrrrGGGrrrrrrrrrGGG")
                self.lastphase[i].append(action)
            else:  # cav
                j= int(rl_id.split("cav_")[ID_IDX])
                self.k.vehicle.apply_acceleration(rl_id, rl_action)
                self.acc[j].append(rl_action)
        self.nowsecond.append((self.nowsecond[-1] + 1))
