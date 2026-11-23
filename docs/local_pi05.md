# Pretrained Pi0.5 policy

The RoboLab-compatible OpenPI fork is installed at `.local/openpi`, with its
own Python 3.11 environment at `.local/openpi/.venv`. RoboLab's environment
contains only the lightweight OpenPI client and its dependencies.

The installed configuration is `pi05_droid_jointpos`, with the pretrained
checkpoint from `gs://openpi-assets-simeval/pi05_droid_jointpos` stored at
`.local/checkpoints/pi05_droid_jointpos` (about 12.4 GB). This is the checkpoint
specified in [RoboLab's Pi0-family instructions](../policies/pi0_family/README.md).
The OpenPI source commit is `aa6420561529593114160d05e5ad155792b272f3` from
<https://github.com/xuningy/openpi>.

## Run a task

You can also launch tasks in the browser: open <http://localhost:8080>, choose
**Tasks → a task → Run with Pi0.5**. Keep the policy server running as described
below. A status banner tracks the episode and links to the completed results.

Start the policy server in one terminal, from the RoboLab repository root:

```bash
bash scripts/local_pi05_server.sh
```

Wait for `server listening on 0.0.0.0:8010`. The server uses physical GPU 1
by default and reserves up to half its memory for JAX. The first inference
request compiles the model and takes longer than subsequent requests.

In another terminal, run a task:

```bash
bash scripts/local_pi05_eval.sh --task BananaInBowlTask
```

To run several tasks, pass their exact names from the dashboard's Tasks tab:

```bash
bash scripts/local_pi05_eval.sh \
  --task BananaInBowlTask RubiksCubeAndBananaTask
```

The evaluation uses GPU 0, one environment and one episode per task. It writes
videos, event logs, HDF5 trajectories and episode results under
`output/<timestamp>_pi05/`. Reload the dashboard at <http://localhost:8080>
and open the new run under Results. The setup verification run is named
`pi05_demo`.

Calling the evaluation script without arguments defaults to `BananaInBowlTask`.
Additional evaluation flags pass through, for example:

```bash
bash scripts/local_pi05_eval.sh --task BananaInBowlTask --num-envs 4
```

These are live model evaluations. Task success depends on the policy and scene;
the policy does not replay the bundled demonstration.

## GPU and port settings

Ports 8000 and 8001 were occupied when this environment was prepared, so both
launch scripts default to 8010. To use another port, set it in both terminals:

```bash
PI05_GPU=2 PI05_PORT=8011 bash scripts/local_pi05_server.sh
PI05_PORT=8011 bash scripts/local_pi05_eval.sh --task BananaInBowlTask
```

The policy server uses OpenPI's default `0.0.0.0` binding. The simulator
connects through `127.0.0.1`.

## Maintenance

The two environments must remain separate. To resync OpenPI:

```bash
source scripts/local_env.sh
cd .local/openpi
CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 GIT_LFS_SKIP_SMUDGE=1 \
  uv sync --locked --python 3.11
```

After resyncing the RoboLab environment with `uv sync`, reinstall the local
OpenPI client, since it is an additional editable install:

```bash
source scripts/local_env.sh
uv pip install --python .venv/bin/python \
  -e .local/openpi/packages/openpi-client
```

Installation and verification logs are in `.runtime/logs/pi05-*.log`.
The OpenPI tokenizer cache is in `.cache/openpi/`.

## Verified on this machine

- OpenPI's isolated dependency check passed.
- Two WebSocket inference requests from the RoboLab environment returned
  finite action arrays of shape `(15, 8)`. The first compiled the model;
  the next completed in about 84 ms with synthetic observations.
- The live `pi05_demo` evaluation completed `BananaInBowlTask` successfully
  after 377 simulation steps, with a subtask score of 1.0. This is one episode,
  not a benchmark success-rate estimate.
- Both the camera and viewport videos, the HDF5 trajectory, event log and
  episode metrics were saved under `output/pi05_demo/` and found by the dashboard.
