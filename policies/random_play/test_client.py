import numpy as np
import pytest

from policies.random_play.client import RandomJointPlayClient


def make_client(seed=7):
    client = RandomJointPlayClient(seed=seed)
    client.set_joint_limits(np.tile([-1.0, 1.0], (7, 1)))
    client.prepare_episode(0, "Unused instruction")
    return client


def test_random_motion_is_bounded_and_records_gripper_actions():
    client = make_client()
    previous = np.full(7, 0.9)
    obs = {"proprio_obs": {"arm_joint_pos": previous[None, :].copy()}}
    gripper_states = set()
    positions = []
    for _ in range(300):
        action = client.infer(obs, "Unused instruction")["action"]
        assert action.shape == (8,)
        assert np.isfinite(action).all()
        assert np.all(action[:7] >= -1) and np.all(action[:7] <= 1)
        assert np.max(np.abs(action[:7] - previous)) <= 0.030001
        assert np.max(np.abs(action[:7] - 0.9)) <= 0.600001
        previous = action[:7].copy()
        positions.append(previous)
        gripper_states.add(action[-1])
    assert gripper_states == {0.0, 1.0}
    assert np.max(np.ptp(positions, axis=0)) > 0.1


def test_seed_is_reproducible_and_parallel_envs_are_independent():
    first, second = make_client(42), make_client(42)
    obs = {"proprio_obs": {"arm_joint_pos": np.zeros((2, 7))}}
    for _ in range(50):
        a = first.infer_batch(obs, "Unused instruction", env_ids=[0, 1])
        b = second.infer_batch(obs, "Unused instruction", env_ids=[1, 0])
        np.testing.assert_array_equal(a[0]["action"], b[0]["action"])
        np.testing.assert_array_equal(a[1]["action"], b[1]["action"])
        assert not np.array_equal(a[0]["action"][:7], a[1]["action"][:7])


@pytest.mark.parametrize("kwargs", [{"seed": -1}, {"joint_radius": 0}, {"target_steps": 0}, {"max_joint_step": 0}])
def test_invalid_motion_settings(kwargs):
    with pytest.raises(ValueError):
        RandomJointPlayClient(**kwargs)
