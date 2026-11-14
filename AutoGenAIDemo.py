from autogen import AssistantAgent, UserProxyAgent
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv(Path(__file__).with_name(".env"))
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(
        "OPENAI_API_KEY is not set. Add it to a .env file beside AutoGenAIDemo.py."
    )

config_list = [
    {
        "model": "gpt-4o-mini",
        "api_key": api_key,
        "temperature": 0.1,
    }
]

assistant = AssistantAgent(
    name="assistant",
    llm_config={"config_list": config_list},
    code_execution_config=False,
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=5,
    code_execution_config=False,
)

user_proxy.initiate_chat(
    assistant,
    message="Analyze an idea and bring up an execution plan to start electric vehicle manufacturing at Tesla.",
)