# asyncio: Python's async library — event loop plus async/await machinery.
import asyncio

# logging: standard library, used only to quiet asyncio's log output.
import logging

# RequirementAgent: BeeAI's experimental step-reasoning agent (same class throughout the series).
from beeai_framework.agents.experimental import RequirementAgent

# UnconstrainedMemory: full-history memory with no trimming.
from beeai_framework.memory import UnconstrainedMemory

# The building blocks for defining a CUSTOM tool — this is the new territory:
#   StringToolOutput - the wrapper type a tool returns (plain-text output)
#   Tool             - the base class you subclass to make your own tool
#   ToolRunOptions   - the per-run options type a tool's _run receives
from beeai_framework.tools import StringToolOutput, Tool, ToolRunOptions

# RunContext: the execution-context object passed into a tool run (carries run-scoped state).
from beeai_framework.context import RunContext

# Emitter: BeeAI's event system. A tool wires itself into it so its activity can be observed.
from beeai_framework.emitter import Emitter

# Chat model class + its config object.
from beeai_framework.backend import ChatModel, ChatModelParameters

# GlobalTrajectoryMiddleware: records the agent's step/tool-call trajectory for observability.
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware

# Pydantic: used to define the tool's typed input schema (same BaseModel/Field idea as before).
from pydantic import BaseModel, Field

# Any: type hint for "any type", used in the constructor signature below.
from typing import Any


# === REAL TOOL CREATION WITH OFFICIAL BEEAI TOOLS ===


# The INPUT SCHEMA for the calculator tool. By subclassing Pydantic's BaseModel, this declares
# exactly what arguments the tool accepts — here, a single string field. The Field description
# is shown to the LLM so it knows how to format the expression it passes in.
class CalculatorInput(BaseModel):
    """Input model for basic mathematical calculations."""
    expression: str = Field(description="Mathematical expression using +, -, *, / (e.g., '10 + 5', '20 - 8', '4 * 6', '15 / 3')")


# The CUSTOM TOOL itself. It subclasses Tool[...] with three type parameters that lock in its
# contract: CalculatorInput (what it takes), ToolRunOptions (its run options), and
# StringToolOutput (what it returns). This generic typing is how BeeAI knows the tool's shape.
class SimpleCalculatorTool(Tool[CalculatorInput, ToolRunOptions, StringToolOutput]):
    """A simple calculator tool for basic arithmetic operations: add, subtract, multiply, divide."""

    # These three class attributes are the tool's identity/metadata that the agent reads:
    name = "SimpleCalculator"                                    # the name the LLM uses to call it
    description = "Performs basic arithmetic calculations: addition (+), subtraction (-), multiplication (*), and division (/)."  # tells the LLM what it does / when to use it
    input_schema = CalculatorInput                              # links the schema class above to this tool

    # Constructor. Accepts an optional options dict and passes it up to the base Tool class.
    def __init__(self, options: dict[str, Any] | None = None) -> None:
        super().__init__(options)

    # Required hook: build the tool's Emitter, which plugs it into BeeAI's event system under
    # a namespace path (["tool", "calculator", "basic"]) so its activity can be tracked/observed.
    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "calculator", "basic"],
            creator=self,
        )

    # A private helper that actually evaluates the math — and, importantly, does so SAFELY.
    def _safe_calculate(self, expression: str) -> float:
        """Safely evaluate basic arithmetic expressions."""
        # Strip spaces so "10 + 5" becomes "10+5" for clean processing.
        expr = expression.replace(' ', '')

        # SECURITY GUARD: define the only characters permitted, then reject anything else.
        # This whitelist is what makes the eval() below safe — no letters, so no function names,
        # no variable access, no code injection can slip through.
        allowed_chars = set('0123456789+-*/().')
        if not all(c in allowed_chars for c in expr):
            # Fail fast with a clear error if a forbidden character appears.
            raise ValueError("Only numbers and basic operators (+, -, *, /, parentheses) are allowed")

        try:
            # eval() runs the arithmetic. The second arg {"__builtins__": {}} strips ALL Python
            # built-in functions from scope, and the empty third arg gives no local names — so
            # eval can do arithmetic and nothing else. (The char whitelist above is the first
            # line of defense; this locked-down environment is the second.)
            result = eval(expr, {"__builtins__": {}}, {})
            return float(result)
        except ZeroDivisionError:
            # Turn Python's divide-by-zero into a friendly domain error.
            raise ValueError("Division by zero is not allowed")
        except Exception as e:
            # Any other parse/eval failure becomes a clear "invalid expression" error.
            raise ValueError(f"Invalid arithmetic expression: {str(e)}")

    # The REQUIRED main method every tool must implement: _run is what executes when the agent
    # calls the tool. It's async (note the await-ability) and receives the parsed input, options,
    # and run context. Returns a StringToolOutput.
    async def _run(
        self, input: CalculatorInput, options: ToolRunOptions | None, context: RunContext
    ) -> StringToolOutput:
        """Perform basic arithmetic calculations."""
        try:
            # input is already a validated CalculatorInput (Pydantic parsed it), so .expression
            # is guaranteed to be a string. Strip stray whitespace.
            expression = input.expression.strip()

            # Do the actual math via the safe helper.
            result = self._safe_calculate(expression)

            # Build a nicely formatted multi-line output string.
            output = f"🧮 Simple Calculator\n"
            output += f"Expression: {expression}\n"
            output += f"Result: {result}\n"

            # Append a human-friendly label guessing the operation type by which symbol appears.
            # (Simple heuristic — checks in order and reports the first operator it finds.)
            if '+' in expression:
                output += "Operation: Addition"
            elif '-' in expression:
                output += "Operation: Subtraction"
            elif '*' in expression:
                output += "Operation: Multiplication"
            elif '/' in expression:
                output += "Operation: Division"
            else:
                output += "Operation: Basic Arithmetic"

            # Wrap the finished string in the tool's output type and return it.
            return StringToolOutput(output)

        except ValueError as e:
            # Expected, validated failures (bad chars, divide-by-zero) — return a clean message
            # rather than crashing the agent. The tool degrades gracefully.
            return StringToolOutput(f"❌ Calculation Error: {str(e)}")
        except Exception as e:
            # Catch-all for anything unforeseen, again returned as output not raised.
            return StringToolOutput(f"❌ Unexpected Error: {str(e)}")


# Async function: an agent that uses our brand-new custom tool.
async def calculator_agent_example():
    """RequirementAgent with SimpleCalculatorTool - Interactive Math Assistant"""

    # Same watsonx Llama 4 Maverick model, temperature 0.
    llm = ChatModel.from_name("watsonx:meta-llama/llama-4-maverick-17b-128e-instruct-fp8", ChatModelParameters(temperature=0))

    # Build the agent — note tools now contains OUR SimpleCalculatorTool(), instantiated.
    calculator_agent = RequirementAgent(
        llm=llm,
        tools=[SimpleCalculatorTool()],            # the custom tool, used exactly like a built-in one
        memory=UnconstrainedMemory(),
        instructions="""You are a helpful math assistant. When users ask for calculations, 
        use the SimpleCalculator tool to provide accurate results. 
        Always show both the expression and the calculated result.""",
        middlewares=[GlobalTrajectoryMiddleware(included=[Tool])],  # observability
        # Note: no requirements this time — the agent uses the tool at its own discretion.
    )

    # A list of natural-language math questions to feed the agent. Note they're phrased
    # conversationally, not as raw expressions — part of the job is the LLM extracting the
    # expression and handing it to the tool in the schema's format.
    math_queries = [
        "What is 15 + 27?",
        "Calculate 144 divided by 12",
        "I need to know what 8 times 9 equals",
        "What's (10 + 5) * 3 - 7?"
    ]

    # Loop over the queries, running the agent on each and printing the exchange.
    for query in math_queries:
        print(f"\n👤 Human: {query}")
        result = await calculator_agent.run(query)
        print(f"🤖 Agent: {result.answer.text}")


# Top-level async entry point.
async def main() -> None:
    # Quiet sub-CRITICAL asyncio logs.
    logging.getLogger('asyncio').setLevel(logging.CRITICAL)

    # Run the demo coroutine.
    await calculator_agent_example()


# Only run when executed directly, not when imported.
if __name__ == "__main__":
    # Start the event loop, run main(), tear it down.
    asyncio.run(main())