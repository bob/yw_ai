
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

from langgraph.checkpoint.memory import InMemorySaver

from langchain_app.model_config import DEFAULT_CHAT_MODEL, chat_model_options
from langchain_app.services.document_fetch import fetch_document_text

@tool
def fetch_text_from_url(url: str) -> str:
    """Fetch a public HTTPS Project Gutenberg document (at most 2 MiB)."""
    return fetch_document_text(url)

SYSTEM_PROMPT = """You are a literary data assistant.

## Capabilities

- `fetch_text_from_url`: loads document text from a URL into the conversation.
Do not guess line counts or positions—ground them in tool results from the saved file."""

content = """Project Gutenberg hosts a full plain-text copy of F. Scott Fitzgerald's The Great Gatsby.
URL: https://www.gutenberg.org/files/64317/64317-0.txt

Answer as much as you can:

1) How many lines in the complete Gutenberg file contain the substring `Gatsby` (count lines, not occurrences within a line, each line ends with a line break).
2) The 1-based line number of the first line in the file that contains `Daisy`.
3) A two-sentence neutral synopsis.

Do your best on (1) and (2). If at any point you realize you cannot **verify** an exact answer with
your available tools and reasoning, do not fabricate numbers: use `null` for that field and spell out
the limitation in `how_you_computed_counts`. If you encounter any errors please report what the error was and what the error message was."""

def main() -> None:
    model = init_chat_model(
        DEFAULT_CHAT_MODEL,
        **chat_model_options(DEFAULT_CHAT_MODEL),
        temperature=0.5,
        timeout=300,
        max_tokens=25000,
    )
    agent = create_agent(
        model=model,
        tools=[fetch_text_from_url],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
    )
    print("Running create_agent...", flush=True)
    agent_result = agent.invoke(
        {"messages": [{"role": "user", "content": content}]},
        config={"configurable": {"thread_id": "great-gatsby-lc"}, "recursion_limit": 20},
    )
    print("\ncreate_agent:")
    print(agent_result["messages"][-1].content_blocks)


if __name__ == "__main__":
    main()
