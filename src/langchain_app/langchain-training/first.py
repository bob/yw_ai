from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from langchain_app.model_config import DEFAULT_CHAT_MODEL, chat_model_options


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


def main() -> None:
    agent = create_agent(
        model=init_chat_model(DEFAULT_CHAT_MODEL, **chat_model_options(DEFAULT_CHAT_MODEL)),
        tools=[get_weather],
        system_prompt="You are a helpful assistant",
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
    )
    print(result["messages"][-1].content_blocks)


if __name__ == "__main__":
    main()
