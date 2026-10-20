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
    character_model = "assets/right_hand_muscle_driven.xml",
    motion_file = [os.path.join("motions/right_hand_motions", _) for _ in os.listdir("motions/right_hand_motions")],

    random_init=True,
    future_horizon=4,

    key_link_weights_orient = {
        "R:ulna": .1,
        "R:radius": 0.05,
        "R:lunate": 0.2, 
        
        "R:firstmc": .1,
        "R:proximal_thumb": .1,
        "R:distal_thumb": .1,
        
        "R:proxph2": .1,
        "R:midph2": .1,
        "R:distph2": .1,

        "R:proxph3": .1,
        "R:midph3": .1,
        "R:distph3": .1,
        
        "R:proxph4": .1,
        "R:midph4": .1,
        "R:distph4": .1,
        
        "R:proxph5": .1,
        "R:midph5": .1,
        "R:distph5": .1
    },
    key_link_weights_pos = {
        "R:ulna": .1,
        "R:lunate": .1,
        "R:distal_thumb": .1,
        "R:THtip": .25,
        "R:IFtip": .2,
        "R:MFtip": .2,
        "R:RFtip": .2,
        "R:distph5": .1,
        "R:LFtip": .2
    }
)

training_params = dict(
    max_epochs =    200000,
    save_interval =  20000,
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
