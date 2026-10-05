# AI Agents

An AI agent is a language model wired into a loop of perception, reasoning, and action. Given
a goal, the agent plans, calls tools, observes the results, and replans until the task is done
or it gives up.

## The ReAct pattern

ReAct interleaves reasoning traces with actions: the model writes a thought, takes an action
such as calling a search API, reads the observation, and repeats. Making the reasoning
explicit improves tool-use accuracy because the model commits to a plan before acting.
Modern agent frameworks implement this loop with a scratchpad that persists across steps.

## Tool use

Tools are how agents touch the world: web search, code execution, database queries, calendar
APIs. Each tool needs a precise schema and a clear description, because the model decides
what to call from the description alone. Poorly described tools are the most common cause of
agent failures, ahead of any model limitation.

## Failure modes

Agents fail in recognisable ways. They loop, repeating the same failing action instead of
trying a new approach. They go off-task, pursuing an interesting sub-problem and forgetting
the original goal. They misuse tools, passing malformed arguments or ignoring error output.
Guardrails that help include step limits, explicit stop conditions, and a final verification
pass that checks the result against the original request.
