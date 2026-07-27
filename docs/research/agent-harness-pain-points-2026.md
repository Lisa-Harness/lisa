# Pain Points in Agent & Harness Development (2025–2026)

A field report on the five most cited unsolved problems in building and shipping LLM agents, drawn from recent research and engineering practice across OpenAI, Anthropic, deepset, MindStudio, and the emerging agent-observability stack.

______________________________________________________________________

## 1. Evaluation Is the #1 Unsolved Problem

Reproducible agent testing is, by widespread consensus, the single hardest open problem in agent development. Unlike model benchmarking, agent evaluation must account for tool calls, branching trajectories, environment state, and nondeterministic model output — all of which make "run the same test twice and get the same result" nearly impossible in practice.

- **Reproducibility is structurally broken.** Two runs of the same agent on the same task can diverge after the first tool call because of sampling variance, tool latency, or environment drift. Traditional unit-test mentalities do not transfer.
- **Tooling helps but does not solve it.** LangSmith and Galileo provide tracing, scoring, and dataset management, but regressions still slip through because the evaluation harness itself is under-specified.
- **Annotated failure data is scarce.** **AgentErrorBench** provides ~200 annotated failure trajectories, giving researchers a toehold — but it is a drop in the bucket relative to the space of possible agent failures.

### Actionable Takeaways

- Treat evaluation as a first-class engineering artifact, not an afterthought: version your eval datasets, seeds, tool mocks, and environment snapshots together.
- Invest in **trajectory-level assertions**, not just final-answer checks.
- Use AgentErrorBench-style annotated failures as a regression suite seed, then grow your own internal failure corpus.
- Pin model versions, temperatures, and tool schemas in CI; assume any unpinned dependency will eventually break reproducibility.

______________________________________________________________________

## 2. Harness Engineering Is Now the Bottleneck, Not the Model

As frontier models have converged on competent reasoning and tool use, the marginal performance gain from swapping models has shrunk while the marginal gain from improving the **agent harness** — the scaffolding, prompt routing, tool selection, retry logic, and context management around the model — has grown.

- **Both OpenAI and Anthropic have published on this.** Their guidance increasingly emphasizes harness design: when to use sub-agents, how to structure tool descriptions, how to manage context windows, and how to decompose tasks.
- **deepset demonstrated the magnitude.** Harness-only changes — without touching the underlying model — moved agents **20+ ranking positions** on agent benchmarks. The harness is where leverage lives.
- **Harness vs. framework is a real distinction (MindStudio).** A *framework* is a reusable library of agent patterns; a *harness* is the bespoke runtime wrapping a specific model for a specific workflow. Conflating them leads to over-reliance on generic frameworks when custom harness tuning is what actually moves the needle.

### Actionable Takeaways

- Budget harness engineering as a core deliverable, not glue work.
- Measure harness changes in isolation (A/B against the same model) before attributing gains to model upgrades.
- Distinguish "framework adoption" from "harness design": use frameworks for boilerplate, but own the harness logic that differentiates your agent.
- Instrument tool-call success rates, retry counts, and context-utilization metrics as harness health signals.

______________________________________________________________________

## 3. Long-Running Agent Memory and Session Continuity

Most production agents are **stateless across sessions**: each new conversation starts with no memory of prior runs, prior user preferences, or prior tool outcomes. This is a fundamental limit on agents that are supposed to operate over days, weeks, or evolving projects.

- **Every session starts at zero.** Without explicit memory infrastructure, agents re-derive context, re-ask clarifying questions, and re-make mistakes they already made yesterday.
- **Anthropic's shift-engineer analogy.** A long-running agent is like a new engineer joining each shift with no handoff notes: capable, but perpetually cold-started. The cost is not just inefficiency — it is broken trust.
- **Context handoff is an open problem.** How much of the prior session should be summarized, persisted, and re-injected? Too little and the agent forgets; too much and the context window drowns. There is no consensus schema or protocol.

### Actionable Takeaways

- Design an explicit **memory layer**: what is persisted (facts, decisions, tool outcomes), for how long, and how it is retrieved.
- Prefer structured memory records over dumping raw transcripts back into context.
- Build a "handoff document" at session end — a compact, agent-readable summary — and load it at session start.
- Treat memory retrieval as its own eval surface: stale or irrelevant memory is often worse than no memory.

______________________________________________________________________

## 4. Debugging Multi-Agent Systems Is Extremely Difficult

Multi-agent architectures compound the observability problem. Failures are silent, loops are self-reinforcing, and the volume of trace data makes manual review impractical.

- **Silent tool failures.** A tool returns an empty or malformed result; the agent interprets it as success and proceeds, producing confidently wrong downstream behavior.
- **Hallucination loops.** Agents can enter cycles where they hallucinate a tool's output, act on it, fail, and re-hallucinate — with no external signal breaking the loop.
- **Manual review doesn't scale.** Reading trace trees by hand is impractical for any non-trivial agent run; the signal-to-noise ratio is too low.
- **Research is emerging.** **LADYBUG** and **AgentErrorBench** target exactly this gap, offering structured approaches to identifying, classifying, and reproducing agent errors across multi-step trajectories.

### Actionable Takeaways

- Log every tool call with full input/output and a success/failure flag; never let a tool fail silently upstream.
- Add **loop detectors**: cap consecutive retries, detect repeated tool-call signatures, and break cycles with a forced re-plan.
- Use LADYBUG/AgentErrorBench-style error taxonomies to tag failures in your traces so they become searchable.
- Build automated "trace diffing" so a regression run can be compared against a known-good trajectory rather than read end-to-end.

______________________________________________________________________

## 5. The Last Mile Problem

Agents routinely reach ~80% correct output on complex tasks — and then stall. The remaining 20% requires human cleanup that is often more tedious than doing the task from scratch, because the human must first understand what the agent did, where it went wrong, and how to patch it without breaking the working 80%.

- **80% is not shippable.** The last mile is where perceived value collapses: users do not credit the 80% that worked; they are taxed by the 20% that didn't.
- **More prompting is not the fix.** Diminishing returns on prompt engineering are well documented at this stage; the lever is not "write a better prompt."
- **Structured context is the fix.** The agents that close the last mile do so by giving the human (or a downstream verification agent) **structured, inspectable context** — intermediate state, assumptions, tool outputs, and confidence signals — so cleanup is surgical rather than archaeological.

### Actionable Takeaways

- Design agents to emit **structured artifacts** (JSON state, decision logs, assumption lists) alongside natural-language output.
- Make the 80% inspectable: every claim should trace back to a tool output or source chunk.
- Build a "human handoff" output mode that optimizes for *editability*, not just readability.
- Measure last-mile cost directly: time-to-fix per task, not just task success rate.

______________________________________________________________________

## Alternative Perspectives

### Gateway/Proxy Infrastructure Does Too Much Magic

LLM gateways and proxies (routing, caching, fallback, prompt injection) increasingly hide behavior from the agent developer. When the gateway silently rewrites a request, swaps a model, or strips context, reproducibility and debugging both suffer. The convenience tax is real: magic that helps in week one hurts in month six.

### Harness vs. Framework Maintenance Burden

The harness/framework distinction cuts both ways. Frameworks reduce boilerplate but impose an upgrade treadmill and abstraction leakage; custom harnesses give control but create a maintenance surface that most teams underestimate. The honest framing is not "frameworks bad, harnesses good" — it is that **the harness is where your competitive advantage and your technical debt both accumulate**, and teams rarely budget for the latter.

### Enterprise Framing Gap

Most agent research and tooling is built for the developer-in-a-Git-repo persona. Enterprise buyers operate under constraints the literature largely ignores: procurement, data residency, audit trails, identity propagation through tool calls, and SLA obligations. The pain points above are real, but in enterprise contexts they are downstream of a more basic gap: **agents are not framed, sold, or governed as production systems**, so the engineering investment in evaluation, memory, and debugging never gets funded at the right level.

______________________________________________________________________

## Sources

- OpenAI — Building Effective Agents (practical guidance on harness/scaffolding design): <https://platform.openai.com/docs/guides/agents>
- Anthropic — Building Effective Agents (multi-agent patterns, context management, shift-engineer analogy): <https://www.anthropic.com/research/building-effective-agents>
- deepset — Agent harness impact on benchmark rankings (harness-only changes moving 20+ positions): <https://deepset.ai/>
- MindStudio — Harness vs. framework distinction in agent architecture: <https://mindstudio.ai/>
- LangSmith — Agent tracing, evaluation, and dataset management: <https://docs.smith.langchain.com/>
- Galileo — Agent observability and evaluation platform: <https://www.galileo.ai/>
- AgentErrorBench — Annotated agent failure trajectories (~200 cases): <https://github.com/AgentErrorBench/AgentErrorBench>
- LADYBUG — Debugging and error classification for multi-agent systems: <https://github.com/ladybug-project/ladybug>

______________________________________________________________________

*Document status: field snapshot, 2025–2026. Pain points are ordered by consensus severity, not by novelty.*

<!-- generated by lisa · provider: ollama · model: glm-5.2:cloud · 2026-07-27 -->
