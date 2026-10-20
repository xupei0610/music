import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".2"

env_cls = "MuscleTracking"
env_params = dict(
    fps = 480,
    contactable_links = None,
    episode_length = 480*3,
    frameskip = 1,
    control_freq = 8,
    character_model = "assets/left_hand_muscle_driven.xml",
    motion_file = [os.path.join("motions/left_hand_motions", _) for _ in os.listdir("motions/left_hand_motions")],

    random_init=True,
    future_horizon=4,

    importance_sampling = 20,
    importance_discount = 0.99,
    importance_decay = 0.5,
    importance_scale = 5,

    key_link_weights_orient = {
        "L:ulna": .1,
        "L:radius": 0.05,
        "L:lunate": 0.2, 
        
        "L:firstmc": .1,
        "L:proximal_thumb": .1,
        "L:distal_thumb": .1,
        
        "L:proxph2": .1,
        "L:midph2": .1,
        "L:distph2": .1,

        "L:proxph3": .1,
        "L:midph3": .1,
        "L:distph3": .1,
        
        "L:proxph4": .1,
        "L:midph4": .1,
        "L:distph4": .1,
        
        "L:proxph5": .1,
        "L:midph5": .1,
        "L:distph5": .1,
    },
    key_link_weights_pos = {
        "L:ulna": .1,
        "L:lunate": .1,
        "L:distal_thumb": .1,
        "L:THtip": .25,
        "L:IFtip": .2,
        "L:MFtip": .2,
        "L:RFtip": .2,
        "L:distph5": .1,
        "L:LFtip": .2
    }
)

training_params = dict(
    max_epochs =    200000,
    save_interval =  500,   # keep ckpt-<epoch> every 500 epochs to roll back to
    horizon = 32,
    num_envs = 8192,
    terminate_reward = 0,
    reg = 10,   # penalty on action stds at the max_sigma cap (see main.py)
)

model_params = dict(
    init_sigma = 0.1,
    max_sigma = 0.5,
    normalizer_scale = 3,
    latent_dim = 32,
    normalize_latent = True,
    normalize_value = False
)

discriminators = {}
