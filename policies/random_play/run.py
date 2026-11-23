"""Record short random interactions using RoboLab's normal evaluation outputs."""
# isort: skip_file

import argparse
import json
from pathlib import Path
import sys
import traceback

import cv2  # noqa: F401  must be imported before isaaclab
from isaaclab.app import AppLauncher
from robolab.eval.runner import add_common_eval_args, run_evaluation

parser = argparse.ArgumentParser(description=__doc__)
add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(task=["AnimalsInBinTask"])
parser.add_argument("--num-steps", type=int, default=300)
parser.add_argument("--joint-radius", type=float, default=0.6, help="Random target range about each starting joint position, radians.")
parser.add_argument("--target-steps", type=int, default=30, help="Steps between random arm/gripper targets.")
parser.add_argument("--max-joint-step", type=float, default=0.03, help="Maximum joint-command change per step, radians.")
args, _ = parser.parse_known_args()
if args.num_steps < 1 or args.joint_radius <= 0 or args.target_steps < 1 or args.max_joint_step <= 0:
    parser.error("Random-play step counts and motion parameters must be positive.")
args.enable_cameras = True
launcher = AppLauncher(args)
simulation_app = launcher.app

import numpy as np
import robolab.constants
from policies.random_play.client import RandomJointPlayClient
from robolab.constants import get_output_dir
from robolab.core.utils.isaaclab_compat import as_torch, env_cfg_to_recording_dict
from robolab.eval.episode import run_episode
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs

robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = args.enable_subtask
robolab.constants.RECORD_IMAGE_DATA = False


def make_client(_args):
    return RandomJointPlayClient(seed=args.seed, joint_radius=args.joint_radius,
                                 target_steps=args.target_steps, max_joint_step=args.max_joint_step)


def random_episode(env, env_cfg, client, **kwargs):
    # max_episode_length is derived from this duration in Isaac Lab. Configure
    # the normal timeout, so automatic HDF5 export and termination still run.
    duration = float(np.nextafter(args.num_steps * env.step_dt, 0.0))
    env.cfg.episode_length_s = duration
    env_cfg.episode_length_s = duration
    robot = env.scene["robot"]
    indices = [robot.data.joint_names.index(f"panda_joint{i}") for i in range(1, 8)]
    limits = as_torch(robot.data.soft_joint_pos_limits)[0, indices].detach().cpu().numpy()
    client.set_joint_limits(limits)
    settings = {"mode": "random_joint_play", "seed": args.seed, "num_steps": args.num_steps,
                "joint_radius": args.joint_radius, "target_steps": args.target_steps,
                "max_joint_step": args.max_joint_step}
    env_cfg.random_play = settings
    path = Path(get_output_dir()) / "env_cfg.json"
    path.write_text(json.dumps(env_cfg_to_recording_dict(env_cfg), default=str))
    results, messages, timing = run_episode(env=env, env_cfg=env_cfg, client=client, **kwargs)
    for result in results:
        result["episode_metadata"] = {"interaction_mode": "random_joint_play", "random_play": settings}
    return results, messages, timing


if __name__ == "__main__":
    try:
        auto_register_droid_envs(task_dirs=args.task_dirs, task=args.task)
        run_evaluation(args, policy="random_joint_play", client_factory=make_client, episode_runner=random_episode)
    except Exception:
        traceback.print_exc()
        # Some Isaac Sim versions exit(0) in close(); retain a real failure code.
        sys.stdout.flush()
        sys.stderr.flush()
        try:
            simulation_app.close()
        finally:
            sys.exit(1)
    else:
        simulation_app.close()
