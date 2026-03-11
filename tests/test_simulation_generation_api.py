import inspect

from evacmob.simulation_generation import generate_random_trajectories


def test_generation_api_exposes_three_distance_knobs():
    params = inspect.signature(generate_random_trajectories).parameters
    assert "near_scale_m" in params
    assert "far_scale_m" in params
    assert "evac_min_dist_m" in params
