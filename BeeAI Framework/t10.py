# asyncio: Python's async library — event loop plus async/await machinery.
import asyncio

# logging: standard library, used only to quiet asyncio's log output.
import logging

# RequirementAgent: BeeAI's experimental step-reasoning agent (same class throughout).
from beeai_framework.agents.experimental import RequirementAgent

# ConditionalRequirement: rule object for constraining tool order/counts.
from beeai_framework.agents.experimental.requirements.conditional import ConditionalRequirement

# AskPermissionRequirement: the NEW requirement type. Instead of an automatic rule, it pauses
# execution and asks a human to approve a tool call before it runs — human-in-the-loop control.
from beeai_framework.agents.experimental.requirements.ask_permission import AskPermissionRequirement

# UnconstrainedMemory: full-history memory with no trimming.
from beeai_framework.memory import UnconstrainedMemory

# Chat model class + its config object.
from beeai_framework.backend import ChatModel, ChatModelParameters

# ThinkTool: explicit deliberate-reasoning tool.
from beeai_framework.tools.think import ThinkTool

# WikipediaTool: research tool (live Wikipedia lookups) — the "external access" being gated here.
from beeai_framework.tools.search.wikipedia import WikipediaTool

# GlobalTrajectoryMiddleware: records the agent's step/tool-call trajectory for observability.
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware

# Tool: base class, used as the middleware's type filter.
from beeai_framework.tools import Tool


# Async function: the full production pattern — reasoning control PLUS a human approval gate.
async def production_security_example():
    """
    Production-Ready RequirementAgent with Security Approval
    
    AskPermissionRequirement adds human-in-the-loop security controls.
    Same query, same tracking - but now with approval workflow.
    """
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
    secure_agent = RequirementAgent(
        llm=llm,                                   # the reasoning model
        tools=[ThinkTool(), WikipediaTool()],      # thinking + research
        memory=UnconstrainedMemory(),              # full conversation memory
        instructions=SYSTEM_INSTRUCTIONS,          # persona/methodology
        middlewares=[GlobalTrajectoryMiddleware(included=[Tool])],  # observability

        # Three requirements now work together as a layered pipeline around the WikipediaTool:
        requirements=[
            # Layer 1 — thinking control (familiar from earlier examples):
            ConditionalRequirement(
                ThinkTool,
                force_at_step=1,            # must think first
                min_invocations=1,          # at least once
                max_invocations=2,          # at most twice
                consecutive_allowed=False   # no back-to-back thinking
            ),

            # Layer 2 — SECURITY GATE (the new piece): before WikipediaTool can run at all,
            # AskPermissionRequirement halts the agent and requests human approval. This is the
            # human-in-the-loop checkpoint for external/network access — nothing leaves the
            # machine until a person says yes.
            AskPermissionRequirement(
                WikipediaTool,
            ),

            # Layer 3 — flow/count control that applies once permission is granted:
            ConditionalRequirement(
                WikipediaTool,
                only_after=[ThinkTool],     # research still must follow a think step
                min_invocations=0,          # now OPTIONAL — floor dropped to 0, so research
                                            # isn't forced (sensible: a human might decline it)
                max_invocations=1           # and capped at a single call even when approved
            )
        ]
    )

    # Same query, constant to isolate the effect of the approval gate.
    ANALYSIS_QUERY = """Analyze the cybersecurity risks of quantum computing for financial institutions. 
    What are the main threats, timeline for concern, and recommended preparation strategies?"""

    # Run the agent. Now expect an interactive pause: when the agent wants to use Wikipedia,
    # it will ask for approval before proceeding, then continue under the layer-3 limits.
    result = await secure_agent.run(ANALYSIS_QUERY)

    # Print the final answer text.
    print(f"\n🛡️ Security-Approved Analysis:\n{result.answer.text}")


# Top-level async entry point.
async def main() -> None:
    # Quiet sub-CRITICAL asyncio logs.
    logging.getLogger('asyncio').setLevel(logging.CRITICAL)

    # Run the demo coroutine.
    await production_security_example()


# Only run when executed directly, not when imported.
if __name__ == "__main__":
    # Start the event loop, run main(), tear it down.
    asyncio.run(main())