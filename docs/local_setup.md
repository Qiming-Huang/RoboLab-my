# Local demo setup

This checkout uses Python 3.11 in `.venv`, with the upstream locked `isaac50`
stack (Isaac Sim 5.0.0 and Isaac Lab 2.2.0). `uv`, Git LFS, Python downloads,
and caches are installed inside the checkout. The system Conda environments
are not used. All Git LFS assets have been downloaded.

## Open the dashboard

From the repository root:

```bash
bash scripts/local_dashboard.sh
```

Open <http://localhost:8080>. The Scenes and Tasks tabs browse the bundled
benchmark. The Results tab reads this checkout's `output/` directory.

If the code runs on a remote server, forward TCP port 8080 in your editor,
or run this on your own computer, substituting the server's SSH address:

```bash
ssh -N -L 8080:127.0.0.1:8080 user@server
```

Then open <http://localhost:8080> on your computer. The launch script binds to
loopback by default. Additional CLI arguments pass through, for example
`bash scripts/local_dashboard.sh --port 8081`.

## Run the robot demonstration

```bash
bash scripts/local_demo.sh
```

This replays the bundled `RubiksCubeAndBananaTask` recording in the actual
simulator with one environment on GPU 0, renders the robot cameras headlessly,
and writes episode results, HDF5 data, event logs, and an MP4 under:

```text
output/playback_recorded_data_RubiksCubeAndBananaTask/
```

Reload the dashboard after the run, select the playback run in Results, then
open its task and episode to play the video. This demonstration replays stored
actions; it needs no policy server or model weights. Evaluating a learned
policy requires the separate server setup described in `policies/README.md`.
Recorded manipulation outcomes can vary with GPU, driver, and simulator
versions; a replay is not a new policy evaluation.

The first simulator launch builds shader caches and can take several minutes.
The local environment script sets `OMNI_KIT_ACCEPT_EULA=Y`, as the upstream
verification suite does, to accept the NVIDIA Omniverse EULA for headless runs.

## Use the environment directly

```bash
source scripts/local_env.sh
python --version
uv sync --locked --extra isaac50 --extra test
python examples/run_empty.py --task BananaInBowlTask --headless --num-steps 50
```

Use the launch scripts or the environment's `python` directly for simulator
commands. If you use `uv run`, retain `--extra isaac50 --extra test` so uv
keeps the simulator stack installed.

Setup and verification logs are under `.runtime/logs/`. Dashboard source
preferences are under `.runtime/config/`, and caches are under `.cache/`.

The repository intentionally overrides Isaac Sim's `websockets==12.0`
requirement with `websockets>=14` for its policy clients. Consequently,
`uv pip check` reports that one upstream dependency mismatch even when the
locked installation is complete. Keep the repository's override when syncing.

## Installation checks

The upstream checks used for this setup can be rerun with:

```bash
source scripts/local_env.sh
python -m pytest tests/test_isaaclab.py tests/test_tasks_valid.py \
  tests/test_registered_envs.py tests/test_run_empty.py -q
```

The bundled replay completed successfully on this machine: 648 simulation
steps, `success: true`, and `score: 1.0`. Its dashboard video, events, and
trajectory endpoints were checked, including HTTP byte-range video playback.
The selected upstream installation checks reached 100% and exited with code
0, including the GPU episode. The Isaac Lab 3-specific check was skipped on
the installed Isaac Lab 2 stack.

## Live Pi0.5 policy

The pretrained Pi0.5 policy is also installed in a separate environment.
See [the Pi0.5 guide](local_pi05.md) for server startup and task selection.

## Recorded random interaction

Use `bash scripts/local_random_play.sh --task AnimalsInBinTask` to record
random arm and gripper motion without a policy server. See
[the random-play guide](local_random_play.md) for settings and dashboard playback.
