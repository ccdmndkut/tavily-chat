import logging
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

# Configure comprehensive logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("tavily-chat.log", mode='a')
    ]
)
logger = logging.getLogger(__name__)
import json
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.graph import CompiledGraph
from pydantic import BaseModel

from backend.chatbot import SimpleChatbot

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting application lifespan...")
    try:
        logger.debug("Initializing MemorySaver checkpointer")
        in_memory_checkpointer = MemorySaver()
        
        logger.debug("Creating SimpleChatbot instance")
        simple_chatbot = SimpleChatbot(checkpointer=in_memory_checkpointer)
        
        logger.debug("Building chatbot graph")
        app.state.agent = simple_chatbot.build_graph()
        
        logger.info("Application startup completed successfully")
        yield
        
    except Exception as e:
        logger.error(f"Error during application startup: {str(e)}", exc_info=True)
        raise
    finally:
        logger.info("Application shutdown completed")


logger.info("Initializing FastAPI application")
app = FastAPI(lifespan=lifespan)

# Configure CORS
vite_app_url = os.getenv("VITE_APP_URL")
logger.debug(f"VITE_APP_URL environment variable: {vite_app_url}")

allowed_origins = [vite_app_url] if vite_app_url else []
logger.info(f"Configuring CORS with allowed origins: {allowed_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_agent():
    logger.debug("Getting agent from app state")
    try:
        agent = app.state.agent
        logger.debug("Successfully retrieved agent from app state")
        return agent
    except Exception as e:
        logger.error(f"Error retrieving agent from app state: {str(e)}", exc_info=True)
        raise


class AgentRequest(BaseModel):
    input: str
    thread_id: str


@app.get("/")
async def ping():
    logger.info("Health check endpoint called")
    return {"message": "Alive"}


@app.post("/stream_agent")
async def stream_agent(
    body: AgentRequest,
    agent: CompiledGraph = Depends(get_agent),
):
    logger.info(f"Stream agent endpoint called with thread_id: {body.thread_id}")
    logger.debug(f"Request input length: {len(body.input)} characters")
    logger.debug(f"Request input preview: {body.input[:100]}...")
    
    try:
        agent_runnable = agent
        logger.debug("Successfully retrieved agent runnable")

        formatted_input = {"messages": [HumanMessage(content=body.input)]}
        logger.debug(f"Formatted input: {formatted_input}")

        async def event_generator():
            config = {"configurable": {"thread_id": body.thread_id}}
            logger.debug(f"Agent config: {config}")
            
            event_count = 0
            try:
                logger.info("Starting agent stream events")
                async for event in agent_runnable.astream_events(
                    input={"messages": formatted_input["messages"]},
                    config=config,
                    version="v1",
                ):
                    event_count += 1
                    kind = event["event"]
                    tags = event.get("tags", [])
                    
                    logger.debug(f"Event #{event_count}: kind={kind}, tags={tags}")

                    if kind == "on_chat_model_stream":
                        content = event["data"]["chunk"].content
                        logger.debug(f"Chat model stream content: {content[:50]}...")
                        if "chatbot" in tags:
                            response_data = {
                                "type": "chatbot",
                                "content": content,
                            }
                            logger.debug(f"Yielding chatbot response: {response_data}")
                            yield (json.dumps(response_data) + "\n")
                    elif kind == "on_llm_stream":
                        # Handle LLM streaming events as well
                        chunk = event["data"]["chunk"]
                        if hasattr(chunk, 'content') and chunk.content:
                            content = chunk.content
                            logger.debug(f"LLM stream content: {content[:50]}...")
                            if "chatbot" in tags:
                                response_data = {
                                    "type": "chatbot",
                                    "content": content,
                                }
                                logger.debug(f"Yielding LLM chatbot response: {response_data}")
                                yield (json.dumps(response_data) + "\n")
                            
                    elif kind == "on_custom_event":
                        event_name = event["name"]
                        if event_name in [
                            "tavily_results",
                            "tavily_status",
                            "router",
                            "auto_tavily_parameters",
                            "chatbot_response",
                        ]:
                            response_data = {
                                "type": event_name,
                                "content": event["data"],
                            }
                            logger.debug(f"Yielding custom event '{event_name}': {str(event['data'])[:100]}...")
                            yield (json.dumps(response_data) + "\n")
                        else:
                            logger.debug(f"Ignoring custom event: {event_name}")
                    elif kind == "on_chain_end":
                        # Check if this is the final response from tavily or chatbot nodes
                        if event.get("name") in ["tavily", "chatbot"] and "output" in event.get("data", {}):
                            output = event["data"]["output"]
                            if "messages" in output and output["messages"]:
                                message = output["messages"][-1]
                                if hasattr(message, 'content'):
                                    content = message.content
                                    logger.debug(f"Final response content: {content[:100]}...")
                                    response_data = {
                                        "type": "chatbot",
                                        "content": content,
                                    }
                                    logger.debug(f"Yielding final response: {response_data}")
                                    yield (json.dumps(response_data) + "\n")
                    else:
                        logger.debug(f"Ignoring event kind: {kind}")
                        
                logger.info(f"Agent stream completed. Total events processed: {event_count}")
                
            except Exception as e:
                logger.error(f"Error in event generator: {str(e)}", exc_info=True)
                # Yield error event to client
                error_response = {
                    "type": "error",
                    "content": f"Stream error: {str(e)}"
                }
                yield (json.dumps(error_response) + "\n")

        logger.info("Returning streaming response")
        return StreamingResponse(event_generator(), media_type="application/json")
        
    except Exception as e:
        logger.error(f"Error in stream_agent endpoint: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    logger.info("Starting Tavily Chat application")
    logger.info("Server configuration: host=0.0.0.0, port=8080")
    
    # Log environment variables (without exposing sensitive data)
    logger.debug("Environment variables check:")
    logger.debug(f"TAVILY_API_KEY: {'SET' if os.getenv('TAVILY_API_KEY') else 'NOT SET'}")
    logger.debug(f"OPENAI_API_KEY: {'SET' if os.getenv('OPENAI_API_KEY') else 'NOT SET'}")
    logger.debug(f"VITE_APP_URL: {os.getenv('VITE_APP_URL', 'NOT SET')}")
    
    try:
        uvicorn.run(app=app, host="0.0.0.0", port=8080)
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}", exc_info=True)
        raise
