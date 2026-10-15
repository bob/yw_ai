from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver

from langchain_app.model_config import DEFAULT_CHAT_MODEL, chat_model_options

@dataclass
class Context:
    user_id: str


agent = create_agent(
    model=init_chat_model(DEFAULT_CHAT_MODEL, **chat_model_options(DEFAULT_CHAT_MODEL)),
    tools=[],
    context_schema=Context,
    checkpointer=InMemorySaver(),
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
    config={"configurable": {"thread_id": str(uuid7())}},
    context=Context(user_id="user-123"),
)
