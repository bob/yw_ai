from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver

from langchain_app.model_config import DEFAULT_CHAT_MODEL, chat_model_options
agent = create_agent(
    model=init_chat_model(DEFAULT_CHAT_MODEL, **chat_model_options(DEFAULT_CHAT_MODEL)),
    tools=[],
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": str(uuid7())}}

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
    config=config,
)

# A follow-up turn on the same conversation: reuse the same thread_id to keep history
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What about tomorrow?"}]},
    config=config,
)
