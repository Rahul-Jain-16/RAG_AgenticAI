# asyncio: Python's async library — event loop plus the async/await machinery.
import asyncio

# logging: standard library, used only to quiet asyncio's log output later.
import logging

# Pydantic's building blocks for defining a typed data schema:
#   BaseModel - the base class you subclass to declare a structured data model
#   Field     - lets you attach metadata (like a description) to each field
from pydantic import BaseModel, Field

# List from typing: used to annotate fields that hold a list of items (e.g. List[str]).
from typing import List

# BeeAI backend pieces: the chat model + its config, plus user/system message types.
from beeai_framework.backend import ChatModel, ChatModelParameters, UserMessage, SystemMessage


# Define the SHAPE we want the LLM's answer to take. By subclassing Pydantic's BaseModel,
# each attribute becomes a typed, validated field. The LLM will be asked to fill this in.
class BusinessPlan(BaseModel):
    """A comprehensive business plan structure."""

    # Each line declares one field: a name, its Python type, and a Field(description=...).
    # The description isn't just a comment — it's sent to the model as guidance for what
    # to put in that field, and Pydantic uses these to build the JSON schema.
    business_name: str = Field(description="Catchy name for the business")
    elevator_pitch: str = Field(description="30-second description of the business")
    target_market: str = Field(description="Primary target audience")
    unique_value_proposition: str = Field(description="What makes this business special")

    # List[str] means this field must be a list of strings, not a single string —
    # so the model knows to return multiple revenue streams, not one blob.
    revenue_streams: List[str] = Field(description="Ways the business will make money")

    startup_costs: str = Field(description="Estimated initial investment needed")
    key_success_factors: List[str] = Field(description="Critical elements for success")


# Async function demonstrating structured (schema-constrained) output.
async def structured_output_example():
    # Load an OpenAI model via BeeAI's provider:model-id syntax. "openai" is the provider
    # prefix (same prefix idea as before), "gpt-5-nano" is the model id. temperature=0 → deterministic.
    llm = ChatModel.from_name("openai:gpt-5-nano", ChatModelParameters(temperature=0))

    # Build the conversation: a system message setting the persona, then the user's request.
    messages = [
        SystemMessage(content="You are an expert business consultant and entrepreneur."),
        UserMessage(content="Create a business plan for a mobile app that helps people find and book unique local experiences in their city.")
    ]

    # The key call: create_structure() instead of plain create(). By passing schema=BusinessPlan,
    # we tell the model "don't just write prose — return data that fits this exact structure."
    # BeeAI handles converting the schema to the model's structured-output format, calling the
    # model, and parsing/validating the result back against BusinessPlan. await waits for it.
    response = await llm.create_structure(
        schema=BusinessPlan,
        messages=messages
    )

    # Echo the request for a readable transcript.
    print("User: Create a business plan for a mobile app that helps people find and book unique local experiences in their city.")
    print("\n🚀 AI-Generated Business Plan:")

    # response.object is the parsed result as a dict, with keys matching the schema fields.
    # Each line pulls one field out by key and prints it.
    print(f"💡 Business Name: {response.object['business_name']}")
    print(f"🎯 Elevator Pitch: {response.object['elevator_pitch']}")
    print(f"👥 Target Market: {response.object['target_market']}")
    print(f"⭐ Unique Value Proposition: {response.object['unique_value_proposition']}")

    # revenue_streams is a list, so ', '.join(...) stitches the items into one comma-separated
    # string for printing on a single line.
    print(f"💰 Revenue Streams: {', '.join(response.object['revenue_streams'])}")

    print(f"💵 Startup Costs: {response.object['startup_costs']}")
    print(f"🔑 Key Success Factors:")

    # key_success_factors is also a list; loop over it and print each on its own bullet line.
    for factor in response.object['key_success_factors']:
        print(f"  - {factor}")


# Top-level async entry point.
async def main() -> None:
    # Quiet sub-CRITICAL asyncio logs to keep the console clean.
    logging.getLogger('asyncio').setLevel(logging.CRITICAL)  # Suppress unwanted warnings

    # Run the demo coroutine to completion.
    await structured_output_example()


# Only run when executed directly, not when imported.
if __name__ == "__main__":
    # Start the event loop, run main(), tear it down cleanly.
    asyncio.run(main())
