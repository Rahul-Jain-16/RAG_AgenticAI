# asyncio: Python's async library — event loop plus async/await machinery.
import asyncio

# logging: standard library, used only to quiet asyncio's log output.
import logging

# RequirementAgent: BeeAI's experimental step-reasoning agent (same class throughout).
from beeai_framework.agents.experimental import RequirementAgent

# ConditionalRequirement: rule object for constraining tool use — and the real star of this
# example, where it's pushed well beyond a simple invocation cap.
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

# Tool: base class, used as a type filter for the middleware.
from beeai_framework.tools import Tool


# Async function: same tools as the last example, but now governed by detailed execution rules.
async def controlled_execution_example():
    """
    RequirementAgent with Controlled Execution - Requirements System
    
    Requirements provide precise control over tool execution order and behavior.
    Same query, same tracking - but now with strict execution rules.
    """
    # Same watsonx Llama 4 Maverick model, temperature 0.
    llm = ChatModel.from_name("watsonx:meta-llama/llama-4-maverick-17b-128e-instruct-fp8", ChatModelParameters(temperature=0))

    # Identical persona/methodology instructions — held constant so the only change versus the
    # previous example is the richer requirements (not the tools, model, or query).
    SYSTEM_INSTRUCTIONS = """You are an expert cybersecurity analyst specializing in threat assessment and risk analysis.

Your methodology:
1. Analyze the threat landscape systematically
2. Research authoritative sources when available
3. Provide comprehensive risk assessment with actionable recommendations
4. Focus on practical, implementable security measures"""

    # Build the agent.
    controlled_agent = RequirementAgent(
        llm=llm,                                   # the reasoning model
        tools=[ThinkTool(), WikipediaTool()],      # same two tools as before
        memory=UnconstrainedMemory(),              # full conversation memory
        instructions=SYSTEM_INSTRUCTIONS,          # persona/methodology
        middlewares=[GlobalTrajectoryMiddleware(included=[Tool])],  # observability

        # REQUIREMENTS: this is where the example levels up. Instead of just capping counts,
        # the rules now dictate ORDER and FLOW of execution — declaratively (you state the
        # rules; the agent's runtime enforces them).
        requirements=[
            # Rule 1 — govern the ThinkTool:
            ConditionalRequirement(
                ThinkTool,
                force_at_step=1,            # the agent MUST call ThinkTool as its very first step
                min_invocations=1,          # it has to think at least once (can't skip thinking)
                max_invocations=3,          # but no more than three times total
                consecutive_allowed=False   # it can't think twice in a row — must do something
                                            # else (e.g. research) between thinking steps
            ),

            # Rule 2 — govern the WikipediaTool:
            ConditionalRequirement(
                WikipediaTool,
                only_after=[ThinkTool],     # research is BLOCKED until at least one think step
                                            # has happened — enforces think-before-research
                min_invocations=1,          # must research at least once (can't answer from
                                            # the model alone)
                max_invocations=2           # at most two research calls
            )
        ]
    )

    # Same query as every prior example, constant to isolate the effect of the new rules.
    ANALYSIS_QUERY = """Analyze the cybersecurity risks of quantum computing for financial institutions. 
    What are the main threats, timeline for concern, and recommended preparation strategies?"""

    # Run the agent. Its tool sequence is now constrained: think first, then (and only then)
    # research, within the count limits — before composing the final answer.
    result = await controlled_agent.run(ANALYSIS_QUERY)

    # Print the final answer text.
    print(f"\n🔧 Controlled Execution Analysis:\n{result.answer.text}")


# Top-level async entry point.
async def main() -> None:
    # Quiet sub-CRITICAL asyncio logs.
    logging.getLogger('asyncio').setLevel(logging.CRITICAL)

    # Run the demo coroutine.
    await controlled_execution_example()


# Only run when executed directly, not when imported.
if __name__ == "__main__":
    # Start the event loop, run main(), tear it down.
    asyncio.run(main())