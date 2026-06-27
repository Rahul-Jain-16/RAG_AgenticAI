# asyncio: Python's async library — event loop plus async/await machinery.
import asyncio

# logging: standard library, used only to quiet asyncio's log output.
import logging

# RequirementAgent: BeeAI's experimental step-reasoning agent (same class throughout the series).
from beeai_framework.agents.experimental import RequirementAgent

# ConditionalRequirement: rule object for constraining tool use (here, capping invocations).
from beeai_framework.agents.experimental.requirements.conditional import ConditionalRequirement

# UnconstrainedMemory: full-history memory with no trimming.
from beeai_framework.memory import UnconstrainedMemory

# Chat model class + its config object.
from beeai_framework.backend import ChatModel, ChatModelParameters

# ThinkTool: the NEW tool this example adds. It gives the agent an explicit "think" step —
# a place to reason/plan deliberately as a tool call, rather than only thinking implicitly.
from beeai_framework.tools.think import ThinkTool

# WikipediaTool: the research tool from the previous example (look up Wikipedia summaries).
from beeai_framework.tools.search.wikipedia import WikipediaTool

# GlobalTrajectoryMiddleware: tracks the agent's step/tool-call trajectory for observability.
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware

# Tool: base class, used below as a type filter for the middleware.
from beeai_framework.tools import Tool


# Async function: same agent, now with BOTH a reasoning tool and a research tool.
async def reasoning_enhanced_agent_example():
    """
    RequirementAgent with Systematic Reasoning - ThinkTool + WikipediaTool
    
    Adding ThinkTool enables structured reasoning alongside research.
    Same query, same tracking - now with visible thinking process.
    """
    # Same watsonx Llama 4 Maverick model, temperature 0.
    llm = ChatModel.from_name("watsonx:meta-llama/llama-4-maverick-17b-128e-instruct-fp8", ChatModelParameters(temperature=0))

    # Identical persona/methodology instructions — held constant so the only change versus the
    # last example is the added ThinkTool.
    SYSTEM_INSTRUCTIONS = """You are an expert cybersecurity analyst specializing in threat assessment and risk analysis.

Your methodology:
1. Analyze the threat landscape systematically
2. Research authoritative sources when available
3. Provide comprehensive risk assessment with actionable recommendations
4. Focus on practical, implementable security measures"""

    # Build the agent.
    reasoning_agent = RequirementAgent(
        llm=llm,                                  # the reasoning model

        # NEW: two tools now. ThinkTool (deliberate reasoning step) + WikipediaTool (research).
        # Order matters as a hint — "think" is listed first, nudging plan-then-research behavior.
        tools=[ThinkTool(), WikipediaTool()],

        memory=UnconstrainedMemory(),             # full conversation memory
        instructions=SYSTEM_INSTRUCTIONS,         # persona/methodology

        # Same observability middleware — record all Tool-type events in the trajectory.
        middlewares=[GlobalTrajectoryMiddleware(included=[Tool])],

        # Requirements now cap BOTH tools at 2 invocations each. Each tool gets its own rule,
        # passed as the class (no parentheses) since the rule targets the tool type.
        requirements=[
            ConditionalRequirement(ThinkTool, max_invocations=2),
            ConditionalRequirement(WikipediaTool, max_invocations=2)
        ]
    )

    # Same query, again constant to isolate the effect of adding the ThinkTool.
    ANALYSIS_QUERY = """Analyze the cybersecurity risks of quantum computing for financial institutions. 
    What are the main threats, timeline for concern, and recommended preparation strategies?"""

    # Run the agent and await the result. The agent may now interleave thinking and research
    # steps (each up to twice) before producing its final answer.
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