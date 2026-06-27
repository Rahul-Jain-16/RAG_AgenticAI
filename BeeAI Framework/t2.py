# Import asyncio: Python's built-in library for writing asynchronous (concurrent) code.
# It provides the event loop and the `async`/`await` machinery this script is built on.
import asyncio

# Import the standard logging library, used here only to silence noisy log output later.
import logging

# Import the pieces we need from the BeeAI framework's backend module:
#   ChatModel            - the main class representing a chat LLM, with a factory for loading models by name
#   ChatModelParameters  - a config object for generation settings (temperature, etc.)
#   UserMessage          - represents a message from the user in the conversation
#   SystemMessage        - represents a system instruction that sets the assistant's behavior
# (Note: ChatModelParameters, UserMessage, and SystemMessage are all used below;
#  there's no unused import here.)
from beeai_framework.backend import ChatModel, ChatModelParameters, UserMessage, SystemMessage


# Define an asynchronous function (note the `async def`) that runs one basic chat exchange.
# Because it's `async`, calling it returns a coroutine that must be awaited to actually run.
async def basic_chat_example():
    # Load a chat model by name using the ChatModel factory method.
    #   "watsonx:ibm/granite-4-h-small" uses a provider:model-id format — "watsonx" is the
    #   provider, "ibm/granite-4-h-small" is the model id (same provider-prefix idea as LiteLLM,
    #   just with a colon instead of a slash as the separator).
    #   ChatModelParameters(temperature=0) sets sampling temperature to 0, making the model's
    #   output as deterministic/focused as possible (least random).
    llm = ChatModel.from_name("watsonx:ibm/granite-4-h-small", ChatModelParameters(temperature=0))

    # Build the conversation as an ordered list of message objects.
    messages = [
        # SystemMessage sets the assistant's role and behavior up front — it's the instruction
        # layer the model reads before responding, not something the end user "says".
        SystemMessage(content="You are a helpful AI assistant and creative writing expert."),

        # UserMessage is the actual user turn — the question/request being sent to the model.
        UserMessage(content="Help me brainstorm a unique business idea for a food delivery service that doesn't exist yet.")
    ]

    # Send the messages to the model and asynchronously wait for the full response.
    #   `await` pauses this coroutine until the network/model call completes, without blocking
    #   the whole program. `create()` is the BeeAI method that performs the generation.
    response = await llm.create(messages=messages)

    # Echo the user's question to the console so the transcript is readable.
    print("User: Help me brainstorm a unique business idea for a food delivery service that doesn't exist yet.")

    # Extract the plain text out of the response object via get_text_content() and print it.
    #   The response is a structured object (it can carry metadata, usage, etc.), so we call a
    #   method to pull just the assistant's text rather than printing the whole object.
    print(f"Assistant: {response.get_text_content()}")

    # Return the full response object to the caller, in case it wants more than just the text.
    return response


# Define the top-level async entry point that orchestrates the run.
# The `-> None` annotation documents that this function doesn't return a meaningful value.
async def main() -> None:
    # Raise asyncio's log level to CRITICAL so lower-severity warnings/info messages are hidden,
    # keeping the console output clean. (Only CRITICAL-level asyncio logs will show.)
    logging.getLogger('asyncio').setLevel(logging.CRITICAL)  # Suppress unwanted warnings

    # Run the chat example and await its result.
    # (The variable is assigned but not used afterward — it's here in case you want to inspect it.)
    response = await basic_chat_example()


# Standard Python idiom: only run the following when this file is executed directly,
# not when it's imported as a module into another file.
if __name__ == "__main__":
    # asyncio.run() starts a fresh event loop, runs the main() coroutine to completion,
    # and then cleanly shuts the loop down. This is the single entry point that kicks
    # off all the `async`/`await` code above.
    asyncio.run(main())