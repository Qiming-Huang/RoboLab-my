# Record random robot interaction

From the repository root:

```bash
bash scripts/local_random_play.sh --task AnimalsInBinTask --num-steps 300
```

This runs one headless episode on GPU 2, with no policy server. At the default
15 Hz control rate, 300 steps gives about 20 simulated seconds. Change the GPU
with `RANDOM_PLAY_GPU=0` when appropriate.

The controller samples random arm targets within 0.6 radians of the starting
joint positions, clips them to the simulated robot's joint limits, and limits
each command change to 0.03 radians per step. It samples a new arm target and
open/closed gripper state every 30 steps. This is random motor exploration;
the controller does not use the task instruction.

Camera and viewport MP4s, HDF5 trajectories, event logs and episode metrics
are written under `output/<timestamp>_random_joint_play/<task>/`. Refresh
<http://localhost:8080>, choose **Results**, and open that experiment, its task,
and its episode to watch the videos. The policy label is `random_joint_play`.
Task success and score still describe the scene's benchmark goal, rather than
whether the random-interaction recording worked.

For a different random sequence or a longer recording:

```bash
bash scripts/local_random_play.sh --task AnimalsInBinTask --seed 42 --num-steps 600
```

Additional controls are `--joint-radius`, `--target-steps`, and
`--max-joint-step`. The selected seed and controller settings are stored in
`env_cfg.json` and the episode result. Raw camera frames are saved as MP4s;
the trajectory recording does not duplicate them as HDF5 image arrays.
