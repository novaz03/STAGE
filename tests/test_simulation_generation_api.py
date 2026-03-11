import inspect

from evacmob.simulation_generation import (
    LABEL_COMPACT_LOCAL,
    LABEL_EXTENSIVE_DISPLACEMENT,
    LABEL_INTERMEDIATE_DIRECTED,
    LEGACY_TO_SEMANTIC_LABEL,
    generate_random_trajectories,
)


def test_label_mapping_matches_semantic_csv_labels():
    assert LEGACY_TO_SEMANTIC_LABEL == {
        "sip_home_grocery": LABEL_COMPACT_LOCAL,
        "sip_hospital": LABEL_INTERMEDIATE_DIRECTED,
        "evac_out_of_zone": LABEL_EXTENSIVE_DISPLACEMENT,
    }


def test_generation_api_exposes_three_distance_knobs():
    params = inspect.signature(generate_random_trajectories).parameters
    assert "near_scale_m" in params
    assert "far_scale_m" in params
    assert "evac_min_dist_m" in params
