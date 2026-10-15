# Extension API

Collab-Overcooked agents can be driven by your own model and observed without changing the
benchmark code. This is how training code (e.g. SFT data collection or RL) plugs into the benchmark.

## Custom LLM backend

`make_agent(mdp, layout, backend=..., ...)` replaces the API request of an agent with a callable:

```python
def backend(messages, call_context):
    """messages: the chat messages of the query (system + user)
    call_context: {"agent_index", "agent", "timestep", "call_index", "call_type", ...}
    returns (text, metadata); metadata may contain "token_count" and anything you need later
    (e.g. token ids and log-probs), it is passed back in the `llm_call` event."""
    text = my_policy.generate(messages)
    return text, {"token_count": len(text.split())}
```

`call_type` is one of:

| call_type | when |
| --- | --- |
| `planner_main` | the regular decision of a timestep |
| `validator_correction` | re-decision after the validator rejected the action |
| `communication` | a reply inside a `Collab(...)` dialogue |
| `format_correction` | re-generation after a missing field or a mixed / malformed Action field |
| `rethink` | the reflection after a successful correction |

`(agent_index, timestep, call_index)` identifies a call; `call_index` counts the calls of an agent within a timestep from 0.
Errors raised by a custom backend are not retried.

## Observer

`make_agent(..., observer=obj)` sends agent events to `obj`. Implement any of the methods below; each receives the agent
as the first argument. A `source` is the `call_context` of the call that produced an action (or `None`).

| method | payload | when |
| --- | --- | --- |
| `on_llm_call(agent, call, prompt, response, metadata)` | `call`: the call context; `metadata`: returned by the backend | after every LLM call |
| `on_call_update(agent, call, fields)` | `call` with its new `call_type`; `fields`, e.g. `forced_action` | a communication call produced an embodied action, or a forced `wait(1)` was assigned to a call |
| `on_action_bound(agent, action, source, primary)` | `primary=True`: the action to execute now; `False`: queued for later | an action produced by a call is selected or queued |
| `on_format_error(agent, source, detail)` | | invalid action name/parameters, missing reply fields, or a mixed / malformed Action field |
| `on_validation(agent, action, source, ok, message, format_error)` | `ok`: the action is valid in the current state | after each validation (an invalid action gets one correction) |
| `on_action_executed(agent, action, source, env_action, interaction_param)` | | an action interacts with the environment (pickup / put / start a utensil, ...) |

Queued actions keep their source, so an action executed several timesteps later is still attributed to the call that
produced it. The source of the action executed in each timestep is also saved in the logs (`content.action_source`).

## Example

```python
from collab_overcooked.agent_factory import make_agent

class Recorder:
    def on_validation(self, agent, action, source, ok, message, format_error):
        print(agent.name, action, "valid" if ok else message, "from call", source and source["call_index"])

chef = make_agent(mdp, "new_env", actor="chef", model="my-policy", backend=backend, observer=Recorder())
```
`tests/test_extension_api.py` runs a complete episode with a backend and an observer.
