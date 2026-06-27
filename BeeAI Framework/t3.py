# asyncio: Python's library for asynchronous code — the event loop and async/await machinery.
import asyncio

# logging: standard library, used here only to silence noisy asyncio log messages.
import logging

# string: Python's string-utilities module. (Note: it's imported but never actually used in
# this script — a candidate for removal under a "no dead code" rule.)
import string

# Pull the needed classes from BeeAI's backend:
#   ChatModel           - the chat LLM class + factory for loading models by name
#   ChatModelParameters - generation config (temperature, etc.)
#   UserMessage         - a user turn in the conversation
# (SystemMessage isn't imported this time because this example only sends a user message.)
from beeai_framework.backend import ChatModel, ChatModelParameters, UserMessage


# Define a small helper class that wraps a text template and fills in variables on demand.
class SimplePromptTemplate:
    """Simple prompt template using Python string formatting."""

    # Constructor: runs when you create an instance. Stores the raw template string on the
    # instance so render() can use it later.
    def __init__(self, template: str):
        # Save the template text as an instance attribute.
        self.template = template

    # render(): takes a dict of variable values and returns the filled-in template string.
    def render(self, variables: dict) -> str:
        """Render the template with provided variables."""
        # Start with a working copy of the raw template. We'll transform it step by step.
        formatted_template = self.template

        # First pass: convert mustache-style {{name}} placeholders into Python's {name} style.
        # We loop over each variable the caller provided and rewrite its placeholder.
        for key, value in variables.items():
            # The f-string f"{{{{{key}}}}}" builds the search target. In an f-string each literal
            # brace is written doubled, so {{{{ → "{{" and }}}} → "}}", with {key} in the middle
            # substituting the variable name. For key="project_name" this produces the string
            # "{{project_name}}" — the mustache placeholder to find.
            # The replacement f"{{{key}}}" builds "{project_name}" — single-brace Python style.
            formatted_template = formatted_template.replace(f"{{{{{key}}}}}", f"{{{key}}}")

        # Second pass: now that placeholders are single-brace {name}, hand the string to Python's
        # str.format(). **variables unpacks the dict into keyword args, so {project_name} etc.
        # get substituted with their actual values. The fully rendered prompt is returned.
        return formatted_template.format(**variables)


# Async function that demonstrates rendering prompts and sending them to the model.
async def prompt_template_example():
    # Load the watsonx Granite model with temperature 0 (deterministic output).
    llm = ChatModel.from_name("watsonx:ibm/granite-4-h-small", ChatModelParameters(temperature=0))

    # The raw template text. A triple-quoted string so it can span multiple lines.
    # Placeholders are written in {{double-brace}} mustache style — these are what render()
    # will swap out per scenario.
    template_content = """
    You are a senior data scientist evaluating a machine learning project proposal.
    
    Project Details:
    - Project Name: {{project_name}}
    - Business Problem: {{business_problem}}
    - Available Data: {{data_description}}
    - Timeline: {{timeline}}
    - Success Metrics: {{success_metrics}}
    
    Please provide:
    1. Feasibility assessment (1-10 scale)
    2. Key technical challenges
    3. Recommended approach
    4. Risk mitigation strategies
    5. Expected outcomes
    
    Be specific and actionable in your recommendations.
    """

    # Wrap the raw text in our template class so we can call .render() on it.
    prompt_template = SimplePromptTemplate(template_content)

    # A list of two scenarios. Each is a dict whose keys exactly match the placeholder names
    # in the template — that matching is what lets render() fill every slot.
    project_scenarios = [
        {
            "project_name": "Smart Inventory Optimization",
            "business_problem": "Reduce inventory costs while maintaining 95% product availability",
            "data_description": "2 years of sales data, supplier lead times, seasonal patterns, 500K records",
            "timeline": "3 months development, 1 month testing",
            "success_metrics": "15% cost reduction, maintain 95% availability, <2% forecast error"
        },
        {
            "project_name": "Fraud Detection System",
            "business_problem": "Detect fraudulent transactions in real-time with minimal false positives",
            "data_description": "1M transaction records, user behavior data, device fingerprints",
            "timeline": "6 months development, 2 months validation",
            "success_metrics": "95% fraud detection rate, <1% false positive rate, <100ms response time"
        }
    ]

    # Loop over the scenarios. enumerate(..., 1) yields (index, scenario) pairs starting the
    # count at 1 instead of 0, so the printed labels read "1" and "2".
    for i, scenario in enumerate(project_scenarios, 1):
        # Print a header. scenario['project_name'] pulls that field for the label.
        # \n adds a blank line above for readability.
        print(f"\n=== Project Evaluation {i}: {scenario['project_name']} ===")

        # Fill the template with this scenario's values, producing the final prompt text.
        rendered_prompt = prompt_template.render(scenario)
        print("\n  Rendered prompt:")
        print(rendered_prompt)

        # Wrap the rendered prompt as a single user message in a list (the API expects a list).
        messages = [UserMessage(content=rendered_prompt)]

        # Send it to the model and await the response (network call — await frees the loop).
        response = await llm.create(messages=messages)

        # Print the model's reply, extracted as plain text.
        print("### LLM response: ###\n")
        print(response.get_text_content())


# Top-level async entry point.
async def main() -> None:
    # Hide sub-CRITICAL asyncio log messages to keep the console clean.
    logging.getLogger('asyncio').setLevel(logging.CRITICAL)  # Suppress unwanted warnings

    # Run the demo coroutine to completion.
    await prompt_template_example()


# Only run when executed directly, not when imported.
if __name__ == "__main__":
    # Start the event loop, run main(), then tear the loop down cleanly.
    asyncio.run(main())