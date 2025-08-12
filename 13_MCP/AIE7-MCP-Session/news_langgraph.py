import asyncio
import os

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, MessagesState, START
from langchain.chat_models import init_chat_model


load_dotenv()


async def build_graph():
    model_name = os.getenv("OPENAI_MODEL", "openai:gpt-4.1-mini")
    model = init_chat_model(model_name)

    client = MultiServerMCPClient(
        {
            "news_server": {
                "command": "python",
                "args": ["./news_server.py"],
                "transport": "stdio",
            }
        }
    )
    tools = await client.get_tools()

    def call_model(state: MessagesState):
        response = model.bind_tools(tools).invoke(state["messages"])
        return {"messages": response}

    builder = StateGraph(MessagesState)
    builder.add_node(call_model)
    builder.add_node(ToolNode(tools))
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", tools_condition)
    builder.add_edge("tools", "call_model")
    return builder.compile()


async def main():
    graph = await build_graph()
    user_prompt = (
        "Find top technology headlines in the US and then search detailed articles about the top company mentioned."
    )
    result = await graph.ainvoke({"messages": user_prompt})
    print("Response:")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())


