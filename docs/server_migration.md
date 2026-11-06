# Server migration

## Code

The development branch is `exp/think-judge` on `origin` (the CookPandar GitHub
repository), not `main` and not the upstream benchmark repository.

```bash
git clone --branch exp/think-judge git@github.com:CookPandar/Collab-Overcooked.git Collab-Overcooked-private
cd Collab-Overcooked-private
python -m venv .venv
source .venv/bin/activate
pip install -e .
export PYTHONPATH="$PWD:$PWD/lib/overcooked_ai${PYTHONPATH:+:$PYTHONPATH}"
```

Install a CUDA-compatible PyTorch and the training dependencies (`peft`,
`pyyaml`, `matplotlib`, `vllm`) for the destination hardware. See
`baselines/README.md`; vLLM may use a separate environment. The transfer
directory's `environment.txt` records installed source-server versions, not a
portable lockfile. Do not blindly reuse CUDA wheels from another machine.

`runs/think_judge/` contains versioned experiment scripts/configs and analysis
tools. Its generated outputs, logs, adapters and checkpoints remain ignored.
Historical launchers assume `~/Collab-Overcooked-private`,
`~/collab_overcooked/bin/python`, `~/vllm/bin/python` and lab-specific GPUs.
Review paths and GPU selection before using them. Prefer direct module commands
with destination-specific overrides:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 -m baselines.rl.train \
  --config baselines/configs/rl.yaml \
  model=/new/path/to/Qwen3-1.7B \
  'adapters={chef: /new/path/to/Chef, assistant: /new/path/to/Assistant}' \
  vllm_python=/new/path/to/vllm/bin/python output_dir=runs/new_run
```

## Claude history

The private export includes both this project's current and earlier checkout's
transcripts, subagent transcripts, project memory, matching file-history/tasks,
and project-filtered command history. It does not include other projects,
credentials, global settings, environment dumps, or shell snapshots. Historical
messages and tool output can still contain private information or secrets: use
private SCP/rsync, never commit or publicly upload this archive.

On the destination, install/authenticate Claude separately. Stop Claude before
restoring; back up any existing `~/.claude` first. Extract the trusted archive
into a private directory, then run (substitute the actual paths):

```bash
umask 077
mkdir -p ~/migration-restore
tar -xzf /path/to/claude-history.tar.gz -C ~/migration-restore
python scripts/claude_history.py restore \
  --source ~/migration-restore/claude-history \
  --project "$PWD"
claude --resume
```

The helper merges both old project histories into the new project's path key,
rebases structured working-directory metadata, and refuses to overwrite
conflicting existing files. Original dialogue, tool arguments, file-snapshot
paths and memory text retain historical absolute paths; tell the resumed Claude
the new repository/model/data locations. Do not replay historical file rollback
operations on the new checkout. Restoration is a local-file migration, not a
guarantee that every Claude version accepts every old session format. Session
UUIDs are in `manifest.json` and can also be passed to `claude --resume UUID`.
No remote model call is needed to export/restore.

To export again (avoid editing the same sessions during the snapshot):

```bash
python scripts/claude_history.py export \
  --project "$PWD" \
  --project /home/zhangshuwen/Collab-Overcooked \
  --output /private/transfer/claude-history
tar -czf /private/transfer/claude-history.tar.gz -C /private/transfer claude-history
```

## Models And Experiment State

Git and the Claude archive do **not** contain training data or model weights.
`runs/` on the source server contains absolute symlinks into
`/archive/public/zhangshuwen/`; copying the symlinks alone will not transfer data.
The transfer directory's `external-links.tsv` records their exact targets.

For continuing the old experiments, selectively copy the relevant datasets,
SFT adapters and RL checkpoints with `rsync -aL` (dereference symlinks), for example:

```bash
rsync -aL --info=progress2 OLD_SERVER:/home/zhangshuwen/Collab-Overcooked-private/runs/think_judge/sft_think_l12/ runs/think_judge/sft_think_l12/
rsync -aL --info=progress2 OLD_SERVER:/home/zhangshuwen/Collab-Overcooked-private/runs/think_judge/data/ runs/think_judge/data/
# Select the experiment(s) you actually need, rather than copying all runs.
rsync -aL --info=progress2 OLD_SERVER:/home/zhangshuwen/Collab-Overcooked-private/runs/think_judge/rl/EXPERIMENT/ runs/think_judge/rl/EXPERIMENT/
```

Copy/re-download the base model, and update `model`, `adapters`, `vllm_python`,
judge endpoints/credential-file paths and output paths in the chosen config.
To resume, retain the experiment's `state.json`, checkpoint directories and
config; inspect `baselines/README.md` for the trainer's resume behavior.
API credentials must be configured separately on the new server.

The private `local-extras.tar.gz` preserves local logs/symlinks and the untracked
`temp/` research material as a precaution, without dereferencing external data.
Extract it separately and inspect before merging into a checkout. The Git bundle
is an offline fallback for this development branch; the normal path is Git clone.
