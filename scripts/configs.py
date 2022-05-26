dataset_args = {
    "path": "data/2dt_heart.mat",
    "split_ratio": 0.9,
    "seed": 0,
    "half_window_size": 5,  # same as hands-on 8
    "sampling_ratio": 0.3,  # same as hands-on 8
    "noise_std": 0.01,
    "eps_log": 1e-6
}

activation_params = {
    "min_val": -1,
    "max_val": 1
}

num_knots = 31
