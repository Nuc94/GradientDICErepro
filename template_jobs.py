from examples import *

import json #line added by nuc to retrieve what was already done
import os #line added in order to effectively list directories available
import time #to generate seeds

import metadata_handle

def batch_atari():
    cf = Config()
    cf.add_argument('--i', type=int, default=0)
    cf.add_argument('--j', type=int, default=0)
    cf.merge()

    games = [
        'BreakoutNoFrameskip-v4',
        # 'AlienNoFrameskip-v4',
        # 'DemonAttackNoFrameskip-v4',
        # 'SeaquestNoFrameskip-v4',
        # 'MsPacmanNoFrameskip-v4'
    ]

    algos = [
        dqn_pixel,
        quantile_regression_dqn_pixel,
        categorical_dqn_pixel,
        a2c_pixel,
        n_step_dqn_pixel,
        option_critic_pixel,
        ppo_pixel,
    ]

    algo = algos[cf.i]

    for game in games:
        for r in range(1):
            algo(game=game, run=r, remark=algo.__name__)

    exit()


def batch_mujoco():
    cf = Config()
    cf.add_argument('--i', type=int, default=0)
    cf.add_argument('--j', type=int, default=0)
    cf.merge()

    games = [
        'dm-acrobot-swingup',
        'dm-acrobot-swingup_sparse',
        'dm-ball_in_cup-catch',
        'dm-cartpole-swingup',
        'dm-cartpole-swingup_sparse',
        'dm-cartpole-balance',
        'dm-cartpole-balance_sparse',
        'dm-cheetah-run',
        'dm-finger-turn_hard',
        'dm-finger-spin',
        'dm-finger-turn_easy',
        'dm-fish-upright',
        'dm-fish-swim',
        'dm-hopper-stand',
        'dm-hopper-hop',
        'dm-humanoid-stand',
        'dm-humanoid-walk',
        'dm-humanoid-run',
        'dm-manipulator-bring_ball',
        'dm-pendulum-swingup',
        'dm-point_mass-easy',
        'dm-reacher-easy',
        'dm-reacher-hard',
        'dm-swimmer-swimmer15',
        'dm-swimmer-swimmer6',
        'dm-walker-stand',
        'dm-walker-walk',
        'dm-walker-run',
    ]

    # games = ['HalfCheetah-v2', 'Walker2d-v2', 'Swimmer-v2', 'Hopper-v2', 'Reacher-v2']
    # games = ['HalfCheetah-v2', 'Walker2d-v2', 'Swimmer-v2', 'Hopper-v2']
    # games = ['Hopper-v2']
    # games = ['Reacher-v2', 'HalfCheetah-v2']
    games = ['Reacher-v2']

    # lams = dict(GradientDICE=[0.1, 1],
    #             GenDICE=[0.1, 1])

    params = []

    for game in games:
        for algo in [off_policy_evaluation]:
            for discount in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
                for cor in ['GradientDICE', 'GenDICE', 'DualDICE']:
                    for lam in [0.1, 1]:
                        for lr in [1e-2, 5e-3, 1e-3]:
                            # for r in range(0, 18):
                            for r in range(18, 30):
                                params.append([algo, dict(game=game, run=r, discount=discount,
                                                          correction=cor, lr=lr, lam=lam)])

    algo, param = params[cf.i]
    algo(**param)

    exit()


def batch_boyans_chain():
    cf = Config()
    cf.add_argument('--i', type=int, default=0)
    cf.add_argument('--j', type=int, default=0)
    cf.merge()

    games = [
        'BoyansChainTabular-v0',
        'BoyansChainLinear-v0',
    ]
    params = []

    for game in games:
        for algo in ['GenDICE', 'GradientDICE', 'DualDICE']:
            if algo == 'GenDICE':
                activation = 'squared'
            elif algo == 'GradientDICE' or algo == 'DualDICE':
                activation = 'linear'
            else:
                raise NotImplementedError
            for lr in np.power(4.0, np.arange(-6, 0)):
                for r in range(0, 30):
                    for gamma in [0.1, 0.3, 0.5, 0.7, 0.9]:
                        params.append([gradient_dice_boyans_chain,
                                       dict(game=game, algo=algo, lr=lr, discount=gamma, lam=1, run=r,
                                            ridge=0, activation=activation)])
                    for ridge in [0, 0.001, 0.01, 0.1]:
                        params.append([gradient_dice_boyans_chain,
                                       dict(game=game, algo=algo, lr=lr, discount=1, lam=1, run=r,
                                            ridge=ridge, activation=activation)])

    algo, param = params[cf.i]
    algo(**param)

    exit()


def gradient_dice_boyans_chain(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('lam', 1)
    kwargs.setdefault('discount', 0.5)
    kwargs.setdefault('lr', 0.001)
    kwargs.setdefault('max_steps', int(3e4))
    kwargs.setdefault('ridge', 0)
    kwargs.setdefault('oracle_dual', False)
    config = Config()
    config.merge(kwargs)

    if config.game == 'BoyansChainTabular-v0':
        config.repr = 'tabular'
    elif config.game == 'BoyansChainLinear-v0':
        config.repr = 'linear'
    else:
        raise NotImplementedError

    config.num_workers = 1
    config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
    config.eval_env = Task(config.game)
    config.optimizer_fn = lambda params: torch.optim.SGD(params, lr=config.lr)
    config.network_fn = lambda: GradientDICENet(
        config.state_dim, config.action_dim, config.activation, config.repr)
    #config.eval_interval = config.max_steps // 100
    config.eval_interval = config.max_steps // 30000
    run_steps(GradientDICE(config))


def td3_correction(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('correction', 'no')
    kwargs.setdefault('lam', 1)
    kwargs.setdefault('activation', 'squared')
    kwargs.setdefault('debug', False)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()
    config.max_steps = int(1e6)
    config.eval_interval = int(1e4)
    config.eval_episodes = 20

    config.network_fn = lambda: TD3Net(
        config.action_dim,
        actor_body_fn=lambda: FCBody(config.state_dim, (400, 300), gate=F.relu),
        critic_body_fn=lambda: FCBody(
            config.state_dim + config.action_dim, (400, 300), gate=F.relu),
        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3),
        critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3))

    batch_size = 100
    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=batch_size)
    config.discount = 0.99
    config.random_process_fn = lambda: GaussianProcess(
        size=(config.action_dim,), std=LinearSchedule(0.1))
    config.td3_noise = 0.2
    config.td3_noise_clip = 0.5
    config.td3_delay = 2
    config.warm_up = (batch_size if config.debug else int(1e4))
    config.target_network_mix = 5e-3

    config.dice_net_fn = lambda: GradientDICEContinuousNet(
        body_tau=FCBody(config.state_dim + config.action_dim),
        body_f=FCBody(config.state_dim + config.action_dim),
        opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3),
        activation=config.activation
    )

    sample_init_env = Task(config.game, num_envs=batch_size)
    config.sample_init_states = lambda: sample_init_env.reset()

    run_steps(TD3CorrectionAgent(config))


def td3_continuous(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('max_steps', int(1e6))
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()
    config.eval_interval = int(1e4)
    config.eval_episodes = 20

    config.network_fn = lambda: TD3Net(
        config.action_dim,
        actor_body_fn=lambda: FCBody(config.state_dim, (400, 300), gate=F.relu),
        critic_body_fn=lambda: FCBody(
            config.state_dim + config.action_dim, (400, 300), gate=F.relu),
        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3),
        critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3))

    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=100)
    config.discount = 0.99
    config.random_process_fn = lambda: GaussianProcess(
        size=(config.action_dim,), std=LinearSchedule(0.1))
    config.td3_noise = 0.2
    config.td3_noise_clip = 0.5
    config.td3_delay = 2
    config.warm_up = int(1e4)
    config.target_network_mix = 5e-3
    run_steps(TD3Agent(config))


def off_policy_evaluation(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('correction', 'no')
    kwargs.setdefault('lam', 1)
    kwargs.setdefault('debug', False)
    kwargs.setdefault('noise_std', 0.05)
    kwargs.setdefault('dataset', 1)
    kwargs.setdefault('discount', None)
    kwargs.setdefault('lr', 0)
    kwargs.setdefault('collect_data', False)
    kwargs.setdefault('target_network_update_freq', 1)
    config = Config()
    config.merge(kwargs)

    if config.correction in ['GradientDICE', 'DualDICE']:
        config.activation = 'linear'
        config.lam = 0.1
    elif config.correction in ['GenDICE']:
        config.activation = 'squared'
        config.lam = 1
    else:
        raise NotImplementedError

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()
    config.max_steps = int(1e3)
    config.eval_interval = config.max_steps // 100

    config.network_fn = lambda: TD3Net(
        config.action_dim,
        actor_body_fn=lambda: FCBody(config.state_dim, (400, 300), gate=F.relu),
        critic_body_fn=lambda: FCBody(
            config.state_dim + config.action_dim, (400, 300), gate=F.relu),
        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3),
        critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3))

    batch_size = 128
    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=batch_size)

    config.dice_net_fn = lambda: GradientDICEContinuousNet(
        body_tau_fn=lambda: FCBody(config.state_dim + config.action_dim, gate=F.relu),
        body_f_fn=lambda: FCBody(config.state_dim + config.action_dim, gate=F.relu),
        opt_fn=lambda params: torch.optim.SGD(params, lr=config.lr),
        activation=config.activation
    )

    sample_init_env = Task(config.game, num_envs=batch_size)
    config.sample_init_states = lambda: sample_init_env.reset()

    if config.collect_data:
        OffPolicyEvaluation(config).collect_data()
    else:
        run_steps(OffPolicyEvaluation(config))

def getSeed():
    now = int(time.time() * 1000)
    now = now % (2**32 - 1)
    return now


if __name__ == '__main__':
    mkdir('log')
    mkdir('data')
    set_one_thread()
    #ok it seems i have to generate a seed
    random_seed()

    select_device(-1)
    #batch_boyans_chain()
    #batch_mujoco()

    # select_device(0)
    # batch_atari()

    with open('to_do.json', 'r') as infile:
        to_do = json.load(infile)

    t_done = False

    while not t_done:
        t_number = np.random.randint(low = 0, high = len(to_do)) % len(to_do)
        task = to_do[t_number]
        if task['runs_required'] > task['runs_done']:
            metadata = task['metadata']
            game = metadata['game']
            algo = metadata['algo']
            lr = metadata['lr']
            discount = metadata['discount']
            ridge = metadata['ridge']

            gradient_dice_boyans_chain(
                # game='BoyansChainTabular-v0',
                game=game,
                algo=algo,
                # algo='GenDICE',
                #algo='DualDICE',
                # ridge=0,
                ridge=ridge,
                discount=discount,
                # activation='squared',
                activation='linear',
                lr=lr,
                log_level=0,
            )

            task['runs_done'] += 1
            t_done = True
            break
    #else:
    #    print('everything done')

    with open('to_do.json', 'w') as outfile:
        json.dump(to_do, outfile)

    game = 'Reacher-v2'
    game = 'BoyansChainTabular-v0'
    algo = 'GradientDICE'
    # td3_continuous(
    #     game=game,
    #     max_steps=int(1e4),
    # )
    '''off_policy_evaluation(
        collect_data=False,
        game=game,
        correction='GradientDICE',
        activation='linear',
        # correction='GenDICE',
        # correction='DualDICE',
        discount=0.1,
        #discount=1,
        lr=1e-2,
        lam=1,
        target_network_update_freq=1,
    )'''

    # td3_correction(
    #     game=game,
    #     # correction='GradientDICE',
    #     correction='GenDICE',
    #     # correction='no',
    #     debug=True,
    # )

    #now nuc will code stuff to re-execute multiple experiments
    #i shall first get insights on what i still need to do
    '''log_dirs = list( os.listdir('tf_log') )
    metadata_count = dict()
    for log_dir in log_dirs:
        log_metadata = metadataFromLogDirName(log_dir)
        metadata_str = metadataToString(log_metadata)
        if metadata_str not in metadata_count.keys():
            metadata_count[ metadata_str ] = 0
        metadata_count[metadata_str] += 1

    n_runs = 10
    lrs = [4 ** -1, 4 ** -2, 4 ** -3,4 ** -4, 4 ** -5, 4 ** -6]
    ridges = [0, 0.1, 0.001, 0.0001]
    discounts = [0.1, 0.3, 0.5, 0.7, 0.9]'''

    #ok some work has been done, and now i need to set everything
    #for the charts

    #this shall be a list of the games i have to execute
    '''games_to_do = [
        {
            'game' : 'BoyansChainTabular-v0',
            'algo' : 'GradientDICE',
            'discount' : 0.1,
            'lr' : 4 ** -4,
            'ridge' : 0.0
        },
        {
            'game' : 'BoyansChainTabular-v0',
            'algo' : 'GradientDICE',
            'discount' : 0.3,
            'lr' : 4 ** -3,
            'ridge' : 0.0
        },
        {
            'game' : 'BoyansChainTabular-v0',
            'algo' : 'GradientDICE',
            'discount' : 0.5,
            'lr' : 4 ** -3,
            'ridge' : 0.0
        },
        {
            'game' : 'BoyansChainTabular-v0',
            'algo' : 'GradientDICE',
            'discount' : 0.7,
            'lr' : 4 ** -2,
            'ridge' : 0.0
        },
        {
            'game' : 'BoyansChainTabular-v0',
            'algo' : 'GradientDICE',
            'discount' : 0.9,
            'lr' : 4 ** -2,
            'ridge' : 0.0
        },
        {
            'game' : 'BoyansChainTabular-v0',
            'algo' : 'GradientDICE',
            'discount' : 1.0,
            'lr' : 4 ** -5,
            'ridge' : 0.0
        },
        {
            'game' : 'BoyansChainTabular-v0',
            'algo' : 'GenDICE',
            'discount' : 0.1,
            'lr' : 4 ** -4,
            'ridge' : 0.0
        },
        {
            'game' : 'BoyansChainTabular-v0',
            'algo' : 'GenDICE',
            'discount' : 0.3,
            'lr' : 4 ** -3,
            'ridge' : 0.0
        },
        {
            'game' : 'BoyansChainTabular-v0',
            'algo' : 'GenDICE',
            'discount' : 0.5,
            'lr' : 4 ** -3,
            'ridge' : 0.0
        },
        {
            'game' : 'BoyansChainTabular-v0',
            'algo' : 'GenDICE',
            'discount' : 0.7,
            'lr' : 4 ** -3,
            'ridge' : 0.0
        },
        {
            'game' : 'BoyansChainTabular-v0',
            'algo' : 'GenDICE',
            'discount' : 0.9,
            'lr' : 4 ** -3,
            'ridge' : 0.0
        },
        {
            'game' : 'BoyansChainTabular-v0',
            'algo' : 'GenDICE',
            'discount' : 1.0,
            'lr' : 4 ** -5,
            'ridge' : 0.0001
        },
        {
            'game' : 'BoyansChainTabular-v0',
            'algo' : 'DualDICE',
            'discount' : 0.1,
            'lr' : 4 ** -3,
            'ridge' : 0.0
        },
        {
            'game' : 'BoyansChainTabular-v0',
            'algo' : 'DualDICE',
            'discount' : 0.3,
            'lr' : 4 ** -5,
            'ridge' : 0.0
        },
        {
            'game' : 'BoyansChainTabular-v0',
            'algo' : 'DualDICE',
            'discount' : 0.5,
            'lr' : 4 ** -5,
            'ridge' : 0.0
        },
        {
            'game' : 'BoyansChainTabular-v0',
            'algo' : 'DualDICE',
            'discount' : 0.7,
            'lr' : 4 ** -5,
            'ridge' : 0.0
        },
        {
            'game' : 'BoyansChainTabular-v0',
            'algo' : 'DualDICE',
            'discount' : 0.9,
            'lr' : 4 ** -6,
            'ridge' : 0.0
        },
        {
            'game' : 'BoyansChainTabular-v0',
            'algo' : 'DualDICE',
            'discount' : 1.0,
            'lr' : 4 ** -3,
            'ridge' : 0.01
        }
    ]'''

    '''games_to_do = [
        {
            'game' : 'BoyansChainLinear-v0',
            'algo' : 'GradientDICE',
            'discount' : 0.1,
            'lr' : 4 ** -4,
            'ridge' : 0.0
        },
        {
            'game' : 'BoyansChainLinear-v0',
            'algo' : 'GradientDICE',
            'discount' : 0.3,
            'lr' : 4 ** -4,
            'ridge' : 0.0
        },
        {
            'game' : 'BoyansChainLinear-v0',
            'algo' : 'GradientDICE',
            'discount' : 0.5,
            'lr' : 4 ** -3,
            'ridge' : 0.0
        },
        {
            'game' : 'BoyansChainLinear-v0',
            'algo' : 'GradientDICE',
            'discount' : 0.7,
            'lr' : 4 ** -3,
            'ridge' : 0.0
        },
        {
            'game' : 'BoyansChainLinear-v0',
            'algo' : 'GradientDICE',
            'discount' : 0.9,
            'lr' : 4 ** -2,
            'ridge' : 0.0
        },
        {
            'game' : 'BoyansChainLinear-v0',
            'algo' : 'GradientDICE',
            'discount' : 1.0,
            'lr' : 4 ** -6,
            'ridge' : 0.0
        },
        {
            'game' : 'BoyansChainLinear-v0',
            'algo' : 'GenDICE',
            'discount' : 0.1,
            'lr' : 4 ** -4,
            'ridge' : 0.0
        },
        {
            'game' : 'BoyansChainLinear-v0',
            'algo' : 'GenDICE',
            'discount' : 0.3,
            'lr' : 4 ** -3,
            'ridge' : 0.0
        },
        {
            'game' : 'BoyansChainLinear-v0',
            'algo' : 'GenDICE',
            'discount' : 0.5,
            'lr' : 4 ** -4,
            'ridge' : 0.0
        },
        {
            'game' : 'BoyansChainLinear-v0',
            'algo' : 'GenDICE',
            'discount' : 0.7,
            'lr' : 4 ** -4,
            'ridge' : 0.0
        },
        {
            'game' : 'BoyansChainLinear-v0',
            'algo' : 'GenDICE',
            'discount' : 0.9,
            'lr' : 4 ** -3,
            'ridge' : 0.0
        },
        {
            'game' : 'BoyansChainTabular-v0',
            'algo' : 'GenDICE',
            'discount' : 1.0,
            'lr' : 4 ** -5,
            'ridge' : 0.001
        },
        {
            'game' : 'BoyansChainLinear-v0',
            'algo' : 'GenDICE',
            'discount' : 1.0,
            'lr' : 4 ** -6,
            'ridge' : 0.001
        },
        {
            'game' : 'BoyansChainLinear-v0',
            'algo' : 'DualDICE',
            'discount' : 0.1,
            'lr' : 4 ** -3,
            'ridge' : 0.0
        },
        {
            'game' : 'BoyansChainLinear-v0',
            'algo' : 'DualDICE',
            'discount' : 0.3,
            'lr' : 4 ** -6,
            'ridge' : 0.0
        },
        {
            'game' : 'BoyansChainLinear-v0',
            'algo' : 'DualDICE',
            'discount' : 0.5,
            'lr' : 4 ** -6,
            'ridge' : 0.0
        },
        {
            'game' : 'BoyansChainLinear-v0',
            'algo' : 'DualDICE',
            'discount' : 0.7,
            'lr' : 4 ** -6,
            'ridge' : 0.0
        },
        {
            'game' : 'BoyansChainLinear-v0',
            'algo' : 'DualDICE',
            'discount' : 0.9,
            'lr' : 4 ** -6,
            'ridge' : 0.0
        },
        {
            'game' : 'BoyansChainTabular-v0',
            'algo' : 'DualDICE',
            'discount' : 1.0,
            'lr' : 4 ** -3,
            'ridge' : 0.1
        },
        {
            'game' : 'BoyansChainLinear-v0',
            'algo' : 'DualDICE',
            'discount' : 1.0,
            'lr' : 4 ** -4,
            'ridge' : 0.1
        }
    ]

    print(metadata_count)

    for run_metadata in games_to_do:
        run_metadata_str = metadataToString( run_metadata )
        if run_metadata_str not in metadata_count.keys():
            print('not found in files!')
            metadata_count[run_metadata_str] = 0
        required_runs = max(0, n_runs - metadata_count[run_metadata_str])
        game = run_metadata['game']
        algo = run_metadata['algo']
        ridge = run_metadata['ridge']
        discount = run_metadata['discount']
        lr = run_metadata['lr']
        print(run_metadata_str)
        print(required_runs)
        for _ in range(required_runs):
            random_seed(getSeed())
            gradient_dice_boyans_chain(
                # game='BoyansChainTabular-v0',
                game=game,
                algo=algo,
                #algo='GenDICE',
                #algo='DualDICE',
                # ridge=0,
                ridge=ridge,
                discount=discount,
                # activation='squared',
                activation='linear',
                lr=lr,
                log_level=0,
            )

    algo = 'GenDICE'
    #that thing was basically done to rerun everything
    for lr in lrs:
        for discount in discounts:
            if discount == 1.0:
                for ridge in ridges:
                    #here i would like to have something that checks how many runs i shall do according to the metadata
                    run_metadata = metadataFromVars(game, algo, discount, lr, ridge)
                    run_metadata_str = metadataToString( run_metadata )
                    if run_metadata_str not in metadata_count.keys():
                        metadata_count[run_metadata_str] = 0
                    required_runs = max(0, n_runs - metadata_count[run_metadata_str])
                    for _ in range(required_runs):
                        random_seed(getSeed())
                        gradient_dice_boyans_chain(
                            # game='BoyansChainTabular-v0',
                            game=game,
                            algo=algo,
                            # algo='GenDICE',
                            #algo='DualDICE',
                            # ridge=0,
                            ridge=ridge,
                            discount=discount,
                            # activation='squared',
                            activation='linear',
                            lr=lr,
                            log_level=0,
                        )
            else:
                ridge = 0
                #here i would like to have something that checks how many runs i shall do according to the metadata
                run_metadata = metadataFromVars(game, algo, discount, lr, ridge)
                run_metadata_str = metadataToString( run_metadata )
                if run_metadata_str not in metadata_count.keys():
                    metadata_count[run_metadata_str] = 0
                required_runs = max(0, n_runs - metadata_count[run_metadata_str])
                for _ in range(required_runs):
                    random_seed(getSeed())
                    gradient_dice_boyans_chain(
                        # game='BoyansChainTabular-v0',
                        game=game,
                        algo=algo,
                        # algo='GenDICE',
                        #algo='DualDICE',
                        # ridge=0,
                        ridge=ridge,
                        discount=discount,
                        # activation='squared',
                        activation='linear',
                        lr=lr,
                        log_level=0,
                    )'''

    
