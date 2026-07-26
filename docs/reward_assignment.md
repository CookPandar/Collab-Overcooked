# Reward Assignment

This document describes the reward signals currently used by the RL and GRPO
pipelines. The authoritative implementation is `ProcessRewardTracker` in
`collab_overcooked/reward/tracker.py`; rollout serialization happens in
`collab_overcooked/training/main_session.py` and
`collab_overcooked/training/mappo_qwen.py`.

## Reward Layers

Each LLM call can produce a transition. The scalar `reward` stored in a rollout
transition is the value used by RL/GRPO updates. The other reward fields are
logged for analysis and should sum to `breakdown_total_reward` when the
transition has a reward breakdown.

Common transition fields:

- `reward`: scalar used by policy optimization.
- `sequence_reward`: executed-action progress reward. This is also stored as
  `process_reward` for backward compatibility.
- `format_reward`: syntax/format reward or penalty.
- `validator_reward`: environment execution validator reward or penalty.
- `communication_reward`: repeated or forced communication penalty.
- `repeat_communication_reward`: repeated-action component of
  `communication_reward`.
- `forced_communication_reward`: forced-reply component of
  `communication_reward`.
- `paired_comm_reward`: request/response communication quality reward.
- `collab_reward`: optional collaborative process reward for useful requests.
- `breakdown_total_reward`: sum of the tracked reward components.
- `reward_source_key`: `(agent_index, source_timestamp, call_index)` used to
  backfill execution rewards to the exact LLM call that generated the action.

## Executed-Action Progress Reward

Field: `sequence_reward`

Default value per successful progress step: `process_progress_reward`, default
`1.0`. The current t3-t13 comm-process GRPO configs set it to `2.0`.

The progress reward is only assigned when an embodied action is actually
executed by the environment. Merely generating an action in an LLM response is
not sufficient. The action must pass format checking, survive validator checks,
be selected as the agent's medium-level action, and be reported by the
environment as the executed `ml_action`.

The tracker maintains one executed-action history per agent and compares that
history against reference demonstrations for the current order. A reward is
given only when the executed action improves the best reference-matching score.
The default metric is `tes`; configs can set `sequence_metric: lcs`.

Important behavior:

- `wait` is ignored by the sequence matcher and receives no progress reward.
- `Collab(...)` actions are communication actions and receive no sequence
  reward.
- Rewards are backfilled to the source LLM call using
  `executed_action_source.call_index`. If the source call cannot be matched,
  the implementation records a mismatch event instead of giving progress credit
  to an unrelated call.
- Snapshot rollouts restore prior executed-action history when available, so
  progress does not restart from zero at snapshot timestep 3.

For `baked_bell_pepper` t3-t13 experiments, typical useful executed actions are
Assistant `place_obj_on_counter()`, Chef `pickup(bell_pepper, counter)`, Chef
`put_obj_in_utensil(oven0)`, and the later bake/serve actions when reachable.

## Format Reward And Penalty

Fields: `format_reward`, `penalties[type=format]`

Configured by:

```yaml
reward:
  format_penalty: 0.2
  format_success_reward: 0.05
```

The implementation stores penalties as negative values, so `format_penalty: 0.2`
means `-0.2`.

Format success reward is given only for rewardable embodied-action calls that:

- have a non-empty action,
- are not `wait`,
- are not `Collab(...)`,
- have a call type that can receive execution reward, currently
  `planner_main` or `validator_correction`,
- did not consume any format penalty.

Format failure suppresses validator reward/penalty for that same call. This is
intentional: if an action cannot be parsed as a valid action format, it should
not also walk through validator scoring.

Examples:

- `Action: place_obj_on_counter()` can receive `+0.05` format reward.
- `Action: put_obj_on_counter()` is a wrong action name and receives format
  penalty.
- More than one embodied action in one response is a format error; only
  communication actions may contain multiple collab sub-actions.

## Validator Reward And Penalty

Fields: `validator_reward`, `penalties[type=validator]`

Configured by:

```yaml
reward:
  validator_penalty: 0.1
  validator_success_reward: 0.05
```

The implementation stores penalties as negative values, so
`validator_penalty: 0.1` means `-0.1`.

Validator success reward is only backfilled when an embodied action is actually
executed successfully by the environment and the source call is rewardable.
Communication-only calls skip validator scoring. `wait` does not receive
validator success reward; this avoids encouraging the model to wait, even though
waiting can be legal in some states.

Validator penalty is for format-correct embodied actions that fail semantic or
environment execution checks, such as trying to put an object into an oven when
the current state does not allow it. If format failed first, validator scoring is
suppressed.

## Repeated And Forced Communication Penalties

Fields: `communication_reward`, `repeat_communication_reward`,
`forced_communication_reward`

Configured by:

```yaml
reward:
  communication_penalty: 0.0
  forced_communication_penalty: 0.05
```

The implementation stores penalties as negative values, so
`forced_communication_penalty: 0.05` means `-0.05`.

Repeated communication compares normalized action signatures per agent. If the
same communication/action signature repeats and the penalty is enabled, the call
receives `communication_penalty`.

Forced communication penalty is used when the environment logic has to force a
reply or repair a communication turn. It is independent of sequence progress.

## Paired Communication Reward

Field: `paired_comm_reward`

Configured by:

```yaml
reward:
  paired_comm_reward_enabled: true
  paired_comm_request_positive_reward: 0.5
  paired_comm_request_negative_reward: -0.1
  paired_comm_response_positive_reward: 0.5
  paired_comm_response_negative_reward: -0.1
  paired_comm_deny_reward: 1.0
```

For the current t3-t13 comm-process configs:

```yaml
reward:
  process_progress_reward: 2.0
  collab_reward_enabled: false
  paired_comm_request_positive_reward: 0.5
  paired_comm_response_positive_reward: 0.5
  paired_comm_deny_reward: 0.5
```

This reward evaluates request-response pairs in communication:

- Initiator reward: a `Collab(request(...))` is helpful if appending the
  requested embodied action to the target agent's action history would improve
  reference progress. Helpful requests receive the positive request reward;
  non-helpful requests receive the negative request reward.
- Responder reward: when a pending request exists, `ack(...)` or the exact
  requested embodied action is rewarded if the request was helpful. Rejecting or
  missing a helpful request is penalized.
- Bad-request denial: if the pending request was not helpful, `deny(...)`
  receives `paired_comm_deny_reward`; following a bad request is penalized.

This signal is independent of whether the requested action is eventually
executed. Actual execution progress is still handled by `sequence_reward`.

## Deprecated Collab Reward

Field: `collab_reward`

YAML switch:

```yaml
reward:
  collab_reward_enabled: true
  sequence_metric: lcs
  sequence_weight: 0.2
```

The current t3-t13 comm-process experiment does not use this signal:

```yaml
reward:
  collab_reward_enabled: false
```

When disabled or absent, `collab_reward` is always `0.0`.

When enabled, the tracker parses `Collab(request(...))` actions. For each
request to the teammate, it temporarily appends the requested embodied actions
to the teammate's executed-action history and recomputes reference progress. If
this improves the teammate's best progress score, the requester receives:

```text
collab_reward = progress_delta * sequence_weight
```

The same useful request is also kept as a pending collaborative execution
request for the receiver. If the receiver later executes that exact requested
embodied action and the execution receives `sequence_reward`, the receiver's
execution-source call receives the same `collab_reward` scale. This encourages
the receiver to obtain useful actions through communication and then actually
execute them, instead of only rewarding the requester.

Receiver-side collaborative reward is gated by execution progress:

- no execution means no receiver-side `collab_reward`,
- executing a different action means no receiver-side `collab_reward`,
- executing the requested action without `sequence_reward` means no
  receiver-side `collab_reward`.

This signal credits the collaborative process around useful embodied actions.
It is separate from `paired_comm_reward`, which scores request/response behavior
such as accepting, denying, or following requests.

The base GRPO t3-t13 configs leave this disabled. The
`*_commprocess.yaml` configs enable it so the communication-process experiment
can be compared against the execution-only baseline.

## Partial Success Reward

Stored in the final selected record's `reward_breakdown.partial_success_reward`
when triggered.

Configured by:

```yaml
trainer:
  partial_success:
    type: oven_cooking
    utensil: oven0
    target: baked_bell_pepper
    terminal_reward: 20.0
    shared_terminal_reward: false
```

For `oven_cooking`, the rollout terminates early when the target item is cooking
in the configured utensil. If the environment has not already returned done, the
last policy record receives `terminal_reward`, default `20.0`.
When `shared_terminal_reward: true`, the same terminal reward is applied once to
each agent's latest transition in the active rollout.

This is a task-level terminal/partial-success reward. It is not the same as
`sequence_reward`; a rollout can reach partial success through environment
dynamics while individual process rewards are still logged separately.

## Team Custom Return

`team_custom_return` in the reward curves is the sum of tracked custom reward
components over completed episodes. It is computed from per-call reward
breakdowns, not directly from the environment return.

Current aggregation includes:

```text
sequence_reward
+ format_reward
+ validator_reward
+ communication_reward
+ paired_comm_reward
```

`collab_reward` is stored in rollouts and per-call breakdowns, but the legacy
episode-level custom-return CSV aggregation currently does not include it unless
that aggregation path is updated. For comm-process experiments, inspect
transition-level `collab_reward` and `breakdown_total_reward` in `.pt` rollouts
or reward audit outputs.

## GRPO Advantage Reward

GRPO uses the transition scalar `reward`, not the raw environment reward alone.
The GRPO implementation computes per-agent reward-to-go over the collected
transition order:

```text
G_t(agent i) = r_t + gamma * G_next(agent i)
```

Default t3-t13 GRPO config:

```yaml
trainer:
  grpo:
    norm_scope: agent
    normalize: mean_std
    gamma: 1.0
    advantage_clip: 10.0
```

The reward-to-go values are normalized separately per agent by distinct return
values. This follows the MARSHAL-style turn-level credit assignment used in this
repo and avoids grouping unrelated trajectories by absolute timestep.

## Practical Debugging Rules

When checking a rollout `.pt` file:

- If an embodied action was generated but not executed, expect no
  `sequence_reward` and no validator success reward.
- If an action has a format penalty, expect validator reward and penalty to be
  `0.0` for that call.
- If `wait` is legal, expect no positive validator reward and no process reward.
- If a communication call proposes a useful teammate action, expect
  `paired_comm_reward` when paired communication is enabled, and expect
  `collab_reward` only when `collab_reward_enabled: true`.
- For process reward bugs, compare `reward_source_key`,
  `metadata.call_index`, `metadata.call_type`, and the executed action source in
  the raw reward entry. Execution credit should go only to the exact call that
  generated the executed action.
