# asyncio: Python's async library — event loop plus async/await machinery.
import asyncio

# logging: standard library, used only to quiet asyncio's log output.
import logging

# RequirementAgent: BeeAI's experimental step-reasoning agent (same class as the last example).
from beeai_framework.agents.experimental import RequirementAgent

# ConditionalRequirement: lets you attach rules/constraints to an agent's tool use —
# here it'll be used to cap how many times a tool may be called.
from beeai_framework.agents.experimental.requirements.conditional import ConditionalRequirement

# UnconstrainedMemory: full-history memory with no trimming (same as before).
from beeai_framework.memory import UnconstrainedMemory

# Chat model class + its config object.
from beeai_framework.backend import ChatModel, ChatModelParameters

# WikipediaTool: a ready-made tool that lets the agent look up Wikipedia summaries —
# this is the new capability that turns "pure LLM" into "research-enhanced."
from beeai_framework.tools.search.wikipedia import WikipediaTool

# GlobalTrajectoryMiddleware: middleware that observes/records the agent's execution
# "trajectory" — the sequence of steps and tool calls it makes — for tracking/debugging.
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware

# Tool: the base class for all tools. Imported so it can be used as a filter (a type) below.
from beeai_framework.tools import Tool


# Async function: the same agent as before, now with a research tool, usage limits, and tracking.
async def wikipedia_enhanced_agent_example():
    """
    RequirementAgent with Wikipedia - Research Enhancement and tracking
    
    Adding WikipediaTool provides access to Wikipedia summaries for contextual research.
    Same query - but now with research capability.
    Moreover, middleware is used to track all tool usage.
    """
    # Same watsonx-hosted Llama 4 Maverick model, temperature 0 (deterministic).
    llm = ChatModel.from_name("watsonx:meta-llama/llama-4-maverick-17b-128e-instruct-fp8", ChatModelParameters(temperature=0))

    # Same persona/methodology instructions as the previous example, kept identical on purpose
    # so the only variable that changed is the tool capability.
    SYSTEM_INSTRUCTIONS = """You are an expert cybersecurity analyst specializing in threat assessment and risk analysis.

Your methodology:
1. Analyze the threat landscape systematically
2. Research authoritative sources when available
3. Provide comprehensive risk assessment with actionable recommendations
4. Focus on practical, implementable security measures"""

    # Build the agent — note the three new arguments versus the minimal example.
    wikipedia_agent = RequirementAgent(
        llm=llm,                                  # the reasoning model
        tools=[WikipediaTool()],                  # NEW: agent can now look things up on Wikipedia
        memory=UnconstrainedMemory(),             # full conversation memory
        instructions=SYSTEM_INSTRUCTIONS,         # persona/methodology

        # NEW: middleware list. GlobalTrajectoryMiddleware(included=[Tool]) attaches a tracker
        # that logs the agent's steps, filtered to just Tool-type events — i.e. record every
        # tool call the agent makes (so you can see when/how Wikipedia was used).
        middlewares=[GlobalTrajectoryMiddleware(included=[Tool])],

        # NEW: requirements list — rules the agent must obey. ConditionalRequirement here says
        # the WikipediaTool may be invoked at most twice (max_invocations=2), preventing the
        # agent from looping on searches endlessly. Note WikipediaTool is passed as the class
        # (no parentheses) — the rule targets the tool type, not a specific instance.
        requirements=[ConditionalRequirement(WikipediaTool, max_invocations=2)]
    )

    # Same query as the previous example, again held constant to isolate the effect of the tool.
    ANALYSIS_QUERY = """Analyze the cybersecurity risks of quantum computing for financial institutions. 
    What are the main threats, timeline for concern, and recommended preparation strategies?"""

    # Run the agent and await the result. Internally the agent may now decide to call Wikipedia
    # (up to twice) before composing its final answer.
    result = await wikipedia_agent.run(ANALYSIS_QUERY)

    # Print the final answer text.
    print(f"\n📖 Research-Enhanced Analysis:\n{result.answer.text}")


# Top-level async entry point.
async def main() -> None:
    # Quiet sub-CRITICAL asyncio logs.
    logging.getLogger('asyncio').setLevel(logging.CRITICAL)

    # Run the demo coroutine.
    await wikipedia_enhanced_agent_example()


# Only run when executed directly, not when imported.
if __name__ == "__main__":
    # Start the event loop, run main(), tear it down.
    asyncio.run(main())