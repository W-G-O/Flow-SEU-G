"""Traffic Light Grid example."""
import json
import ray
from ray.rllib.agents.registry import get_agent_class
from ray.tune import run_experiments
from ray.tune.registry import register_env
from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder
from flow.utils.inflow_methods import get_non_flow_params, get_flow_params
from flow.core.params import SumoParams, EnvParams, InFlows, InitialConfig, NetParams
from flow.core.params import SumoParams, EnvParams, SumoCarFollowingParams
from flow.core.params import VehicleParams
from flow.controllers import SimCarFollowingController
from flow.envs.traffic_light_grid import TrafficLightGridSEUEnv,TrafficLightGridSEUPressEnv,TrafficLightGridSEUYellowEnv,TrafficLightGridSEUYellowPressEnv
from flow.networks.traffic_light_grid import SingleIntersectionNet
from flow.controllers import GridRecycleRouter,GridRouter
import matplotlib.pyplot as plt


# time horizon of a single rollout
HORIZON = 3500
# number of rollouts per training iteration
N_ROLLOUTS = 5
# number of parallel workers
N_CPUS = 10
# N_GPUS = 4
# set to True if you would like to run the experiment with inflows of vehicles
# from the edges, and False otherwise
edge_inflow=200
v_enter = 20
inner_length = 500
long_length = 700
short_length = 500
num_cars_left = 1
num_cars_right = 1
num_cars_top = 1
num_cars_bot = 1
n_columns = 1
n_rows = 1

ADDITIONAL_NET_PARAMS = {
    # dictionary of traffic light grid array data
    "grid_array": {
        # number of horizontal rows of edges
        "row_num": n_rows,
        # number of vertical columns of edges
        "col_num": n_columns,
        # length of inner edges in the traffic light grid network
        "inner_length": inner_length,
        # length of edges where vehicles enter the network
        "short_length": short_length,
        # length of edges where vehicles exit the network
        "long_length": long_length,
        # number of cars starting at the edges heading to the top
        "cars_top": num_cars_top,
        # number of cars starting at the edges heading to the bottom
        "cars_bot": num_cars_bot,
        # number of cars starting at the edges heading to the left
        "cars_left": num_cars_left,
        # number of cars starting at the edges heading to the right
        "cars_right": num_cars_right,
    },
    # number of lanes in the horizontal edges
    "horizontal_lanes": 3,
    # number of lanes in the vertical edges
    "vertical_lanes": 3,
    # speed limit for all edges, may be represented as a float value, or a
    # dictionary with separate values for vertical and horizontal lanes
    "speed_limit": {
        "horizontal": 30,
        "vertical": 30
    },
    "traffic_lights": True
}  # the net_params

tot_cars = ((num_cars_left + num_cars_right) * n_columns
           + (num_cars_top + num_cars_bot) * n_rows)

ADDITIONAL_ENV_PARAMS = {
        'target_velocity': 30,
        'switch_time': 3.0,
        'num_observed': 2,
        'discrete': True,
        'tl_type': 'controlled',
        "max_accel": 3,
        "max_decel": 3,
    }


vehicles = VehicleParams()
vehicles.add(
    veh_id='human',
    acceleration_controller=(SimCarFollowingController, {}),
    car_following_params=SumoCarFollowingParams(
        min_Gap=2.5,
        decel=3.5,  # avoid collisions at emergency stops
        speed_mode="all_checks",
    ),
    routing_controller=(GridRecycleRouter, {}),
    num_vehicles=tot_cars)


# inflows of vehicles are place on all outer edges (listed here)
outer_edges = []
outer_edges += ["left{}_{}".format(n_rows, i) for i in range(n_columns)]
outer_edges += ["right0_{}".format(i) for i in range(n_columns)]
outer_edges += ["bot{}_0".format(i) for i in range(n_rows)]
outer_edges += ["top{}_{}".format(i, n_columns) for i in range(n_rows)]

# equal inflows for each edge (as dictate by the EDGE_INFLOW constant)
inflow = InFlows()
for edge in outer_edges:
    inflow.add(
        veh_type="human",
        edge=edge,
        vehs_per_hour=edge_inflow,
        depart_lane="free",
        depart_speed=15)

flow_params = dict(
    # name of the experiment
    exp_tag='traffic_light_i{}_grid'.format(edge_inflow),

    # name of the flow environment the experiment is running on
    env_name=TrafficLightGridSEUPressEnv,

    # name of the network class the experiment is running on
    network=SingleIntersectionNet,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=1,
        render=False,
        restart_instance=True,
        print_warnings=False,
        emission_path="data/traffic_light_i{}_grid".format(edge_inflow),
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        warmup_steps=500,
        evaluate=False,
        additional_params=ADDITIONAL_ENV_PARAMS,
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component). This is
    # filled in by the setup_exps method below.
    net=NetParams(
        inflows=inflow,
        additional_params=ADDITIONAL_NET_PARAMS,
    ),
    veh=vehicles,
    initial=InitialConfig(
        spacing='custom',
        lanes_distribution=float('inf'),
        shuffle=False,
),
)


def setup_exps():
    """Return the relevant components of an RLlib experiment.

    Returns
    -------
    str
        name of the training algorithm
    str
        name of the gym environment to be trained
    dict
        training configuration parameters
    """
    alg_run = "PPO"

    agent_cls = get_agent_class(alg_run)
    config = agent_cls._default_config.copy()
    config["num_gpus"] = 1
    config["num_workers"] = 10
    config["train_batch_size"] = HORIZON * N_ROLLOUTS
    config["gamma"] = 0.999  # discount rate
    config["model"].update({"fcnet_hiddens": [3, 3]})
    config["use_gae"] = True
    config["lambda"] = 0.97
    config["vf_clip_param"] = 1000
    config["kl_target"] = 0.02
    config["num_sgd_iter"] = 10
    config['clip_actions'] = False  # FIXME(ev) temporary ray bug
    config["horizon"] = HORIZON

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = alg_run

    create_env, gym_name = make_create_env(params=flow_params, version=0)

    # Register as rllib env
    register_env(gym_name, create_env)
    return alg_run, gym_name, config


alg_run, gym_name, config = setup_exps()
ray.init(num_cpus=N_CPUS + 1,num_gpus=1)
trials = run_experiments({
    flow_params["exp_tag"]: {
        "run": alg_run,
        "env": gym_name,
        "config": {
            **config
        },
        "checkpoint_freq": 20,
        "checkpoint_at_end": True,
        "max_failures": 999,
        "stop": {
            "training_iteration": 1000,
        },
    }
})
