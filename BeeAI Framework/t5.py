# asyncio: Python's async library — the event loop plus async/await machinery.
import asyncio

# logging: standard library, used only to silence asyncio's log output later.
import logging

# RequirementAgent: BeeAI's experimental agent class. Unlike a plain ChatModel call, an
# "agent" can reason in steps, follow instructions as a persona, use tools, and keep memory.
from beeai_framework.agents.experimental import RequirementAgent

# UnconstrainedMemory: a memory backend that stores the full conversation history with no
# size/trimming limits — the agent remembers everything in the session.
from beeai_framework.memory import UnconstrainedMemory

# The chat model class and its config object (no message types imported here, because the
# agent manages messages internally rather than you building them by hand).
from beeai_framework.backend import ChatModel, ChatModelParameters


# Async function demonstrating the simplest possible agent: no tools, just an LLM + instructions.
async def minimal_tracked_agent_example():
    """
    Minimal RequirementAgent
    """
    # Load a watsonx-hosted Llama 4 Maverick model. "watsonx:" is the provider prefix,
    # the rest is the model id. temperature=0 → deterministic output.
    llm = ChatModel.from_name("watsonx:meta-llama/llama-4-maverick-17b-128e-instruct-fp8", ChatModelParameters(temperature=0))

    # The system-level instructions defining the agent's persona and methodology. A triple-quoted
    # multi-line string. This is passed to the agent as its standing behavior, not as a user turn.
    SYSTEM_INSTRUCTIONS = """You are an expert cybersecurity analyst specializing in threat assessment and risk analysis.

        Your methodology:
        1. Analyze the threat landscape systematically
        2. Research authoritative sources when available
        3. Provide comprehensive risk assessment with actionable recommendations
        4. Focus on practical, implementable security measures"""

    # Construct the agent by wiring together its parts:
    minimal_agent = RequirementAgent(
        llm=llm,                          # the model that does the reasoning
        tools=[],                         # an empty tool list — this agent can't call any tools yet
        memory=UnconstrainedMemory(),     # a fresh memory store to hold the conversation
        instructions=SYSTEM_INSTRUCTIONS  # the persona/methodology defined above
    )

    # The question to ask. Triple-quoted so it can span two lines.
    ANALYSIS_QUERY = """Analyze the cybersecurity risks of quantum computing for financial institutions. 
    What are the main threats, timeline for concern, and recommended preparation strategies?"""

    # Run the agent on the query and await the result. Note the difference from earlier scripts:
    # you call agent.run(query) with a plain string, and the agent handles assembling messages,
    # applying instructions, and producing an answer — a higher-level interface than llm.create().
    result = await minimal_agent.run(ANALYSIS_QUERY)

    # The agent's final answer lives at result.answer.text. Print it with a header.
    print(f"\n💬 Pure LLM Analysis:\n{result.answer.text}")


# Top-level async entry point.
async def main() -> None:
    # Quiet sub-CRITICAL asyncio logs to keep the console clean.
    logging.getLogger('asyncio').setLevel(logging.CRITICAL)

    # Run the demo coroutine to completion.
    await minimal_tracked_agent_example()


# Only run when executed directly, not when imported.
if __name__ == "__main__":
    # Start the event loop, run main(), tear it down cleanly.
    asyncio.run(main())