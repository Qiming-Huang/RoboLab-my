"""Seeded random joint targets with bounded changes between control steps."""

import numpy as np

from robolab.eval.base_client import InferenceClient


class RandomJointPlayClient(InferenceClient):
    def __init__(self, *, seed=0, joint_radius=0.6, target_steps=30, max_joint_step=0.03):
        super().__init__()
        if seed < 0 or joint_radius <= 0 or target_steps < 1 or max_joint_step <= 0:
            raise ValueError("Use a nonnegative seed and positive random-motion parameters.")
        self.seed = seed
        self.joint_radius = joint_radius
        self.target_steps = target_steps
        self.max_joint_step = max_joint_step
        self._limits = None
        self._motion = {}

    def set_joint_limits(self, limits):
        limits = np.asarray(limits, dtype=np.float32)
        if limits.shape != (7, 2) or not np.isfinite(limits).all() or np.any(limits[:, 0] >= limits[:, 1]):
            raise ValueError("Expected seven finite lower/upper joint limits.")
        self._limits = limits.copy()

    def begin_episode(self, episode_idx):
        super().begin_episode(episode_idx)
        self._motion.clear()

    def _extract_observation(self, raw_obs, *, env_id=0):
        return {"joints": self._to_numpy(raw_obs["proprio_obs"]["arm_joint_pos"], env_id), "env_id": env_id}

    def _pack_request(self, extracted_obs, instruction):
        # Random play deliberately does not read or follow the task instruction.
        return extracted_obs

    def _query_server(self, request):
        if self._limits is None:
            raise RuntimeError("Set the simulated robot's joint limits before random play.")
        lower, upper = self._limits[:, 0], self._limits[:, 1]
        env_id = request["env_id"]
        if env_id not in self._motion:
            initial = np.clip(np.asarray(request["joints"], dtype=np.float32), lower, upper)
            if initial.shape != (7,) or not np.isfinite(initial).all():
                raise ValueError("Expected seven finite observed arm joint positions.")
            self._motion[env_id] = {
                "rng": np.random.default_rng(np.random.SeedSequence([self.seed, self._eval_episode_idx, env_id])),
                "center": initial.copy(), "command": initial.copy(), "step": 0,
            }
        motion = self._motion[env_id]
        if motion["step"] % self.target_steps == 0:
            offset = motion["rng"].uniform(-self.joint_radius, self.joint_radius, size=7)
            motion["target"] = np.clip(motion["center"] + offset, lower, upper)
            motion["gripper"] = float(motion["rng"].integers(0, 2))
        delta = np.clip(motion["target"] - motion["command"], -self.max_joint_step, self.max_joint_step)
        motion["command"] = np.clip(motion["command"] + delta, lower, upper)
        motion["step"] += 1
        return np.r_[motion["command"], motion["gripper"]].astype(np.float32)[None, :]

    def _unpack_response(self, response):
        return response

    def reset(self, *, env_id=None):
        super().reset(env_id=env_id)
        if env_id is None:
            self._motion.clear()
        else:
            self._motion.pop(env_id, None)
