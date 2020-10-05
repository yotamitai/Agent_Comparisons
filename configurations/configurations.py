
from collections import OrderedDict
from frogger.frogger import ACTION_UP_KEY, ACTION_DOWN_KEY, ACTION_LEFT_KEY, ACTION_RIGHT_KEY, HIT_CAR_RWD_ATTR, \
    HIT_WATER_RWD_ATTR, TIME_UP_RWD_ATTR, NEW_LEVEL_RWD_ATTR, FROG_ARRIVED_RWD_ATTR, TICK_RWD_ATTR, NO_LIVES_RWD_ATTR
from interestingness_xrl.scenarios.frogger.configurations import FroggerConfiguration
GAME_GYM_ID = 'Frogger-Custom'


EXPERT_CONFIG = FroggerConfiguration(
    name='expert',
    actions=OrderedDict([
        ('left', ACTION_LEFT_KEY),
        ('right', ACTION_RIGHT_KEY),
        ('up', ACTION_UP_KEY),
        ('down', ACTION_DOWN_KEY),
        # ('nowhere', ACTION_NO_MOVE_KEY)
    ]),
    rewards={
        HIT_CAR_RWD_ATTR: -200.,
        HIT_WATER_RWD_ATTR: -200.,
        TIME_UP_RWD_ATTR: -200.,
        NO_LIVES_RWD_ATTR: -300.,
        NEW_LEVEL_RWD_ATTR: 0.,
        FROG_ARRIVED_RWD_ATTR: 5000.,
        TICK_RWD_ATTR: -1.
    },
    gym_env_id=GAME_GYM_ID,
    lives=3,
    speed=3.,
    level=1,
    num_arrived_frogs=2,
    car_x_vision_num_cells=1.5,
    car_y_vision_num_cells=1.,
    max_steps_life=1000,
    max_steps_per_episode=1000,
    num_episodes=10000,
    num_recorded_videos=10,
    seed=0,
    max_temp=20,
    min_temp=0.05,
    discount=.9,
    learn_rate=.3,
    initial_q_value=5000.
)
