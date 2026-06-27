# asyncio: Python's async library — event loop plus async/await machinery.
import asyncio

# logging: standard library, used only to quiet asyncio's log output.
import logging

# RequirementAgent: BeeAI's experimental step-reasoning agent (same class throughout).
from beeai_framework.agents.experimental import RequirementAgent

# ConditionalRequirement: rule object for constraining tool use — again the focus here.
from beeai_framework.agents.experimental.requirements.conditional import ConditionalRequirement

# UnconstrainedMemory: full-history memory with no trimming.
from beeai_framework.memory import UnconstrainedMemory

# Chat model class + its config object.
from beeai_framework.backend import ChatModel, ChatModelParameters

# ThinkTool: explicit deliberate-reasoning tool.
from beeai_framework.tools.think import ThinkTool

# WikipediaTool: research tool (live Wikipedia lookups).
from beeai_framework.tools.search.wikipedia import WikipediaTool

# GlobalTrajectoryMiddleware: records the agent's step/tool-call trajectory for observability.
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware

# Tool: base class. Used both as a middleware type filter AND — new here — as the target of
# a force_after rule, meaning "any tool" rather than one specific tool.
from beeai_framework.tools import Tool


# Async function: same two tools, but the requirements now create a think-after-everything loop.
async def reasoning_enhanced_agent_example():
    # Same watsonx Llama 4 Maverick model, temperature 0.
    llm = ChatModel.from_name("watsonx:meta-llama/llama-4-maverick-17b-128e-instruct-fp8", ChatModelParameters(temperature=0))

    # Identical persona/methodology instructions, held constant as in every prior example.
    SYSTEM_INSTRUCTIONS = """You are an expert cybersecurity analyst specializing in threat assessment and risk analysis.

Your methodology:
1. Analyze the threat landscape systematically
2. Research authoritative sources when available
3. Provide comprehensive risk assessment with actionable recommendations
4. Focus on practical, implementable security measures"""

    # Build the agent.
    reasoning_agent = RequirementAgent(
        llm=llm,                                   # the reasoning model
        tools=[ThinkTool(), WikipediaTool()],      # thinking + research
        memory=UnconstrainedMemory(),              # full conversation memory
        instructions=SYSTEM_INSTRUCTIONS,          # persona/methodology
        middlewares=[GlobalTrajectoryMiddleware(included=[Tool])],  # observability

        requirements=[
            # A single rule, applied only to ThinkTool, but a powerful one — it weaves a
            # thinking step into every part of the flow:
            ConditionalRequirement(
                ThinkTool,
                force_at_step=1,     # must think first, as the opening move
                force_after=Tool,    # NEW: force a think step after EVERY tool call. Because the
                                     # target is the base class Tool (not a specific tool), this
                                     # means "after any tool runs, think again" — i.e. think →
                                     # tool → think → tool → think... a reflect-after-each-action loop
                min_invocations=1,   # think at least once
                max_invocations=5,   # but cap total thinking at five (the safety valve on the loop)
                consecutive_allowed=False  # no back-to-back thinking — must act between thoughts
            ),
            # The WikipediaTool rule from the previous example is commented out, so research is
            # now UNgoverned: the agent may use Wikipedia freely (or not at all) — no floor, no
            # ceiling, no ordering constraint on it.
            #ConditionalRequirement(WikipediaTool, max_invocations=2)
        ]
    )

    # Same query, constant to isolate the effect of the new force_after rule.
    ANALYSIS_QUERY = """Analyze the cybersecurity risks of quantum computing for financial institutions. 
    What are the main threats, timeline for concern, and recommended preparation strategies?"""

    # Run the agent. Expected shape: think → (some tool) → think → (some tool) → think ...,
    # bounded by max_invocations=5 on thinking, until it composes the final answer.
    result = await reasoning_agent.run(ANALYSIS_QUERY)

    # Print the final answer text.
    print(f"\n🧠 Reasoning + Research Analysis:\n{result.answer.text}")


# Top-level async entry point.
async def main() -> None:
    # Quiet sub-CRITICAL asyncio logs.
    logging.getLogger('asyncio').setLevel(logging.CRITICAL)

    # Run the demo coroutine.
    await reasoning_enhanced_agent_example()


# Only run when executed directly, not when imported.
if __name__ == "__main__":
    # Start the event loop, run main(), tear it down.
    asyncio.run(main())