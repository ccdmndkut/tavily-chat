import logging
import os
from datetime import datetime
from typing import Annotated

from langchain_core.callbacks.manager import dispatch_custom_event
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from tavily import TavilyClient
from typing_extensions import TypedDict

from .prompts import CHATBOT, DEFAULT_SYSTEM_PROMPT, ROUTER, TAVILY
from .utils import format_documents_for_llm, parse_messages, tavily_results_to_documents

# Configure logging for this module
logger = logging.getLogger(__name__)


class State(TypedDict):
    messages: Annotated[list, add_messages]
    search_results: Annotated[list, add_messages]
    response: str


class SimpleChatbot:
    def __init__(
        self,
        model_name: str = "gpt-4o",
        checkpointer: MemorySaver = None,
    ):
        logger.info(f"Initializing SimpleChatbot with model: {model_name}")
        
        try:
            logger.debug("Creating ChatOpenAI instance")
            self.llm = ChatOpenAI(model=model_name)
            
            logger.debug("Setting up chain configurations")
            self.router_chain = ROUTER | self.llm
            self.tavily_chain = TAVILY | self.llm.with_config({"tags": ["chatbot"]})
            self.chatbot_chain = CHATBOT | self.llm.with_config({"tags": ["chatbot"]})
            
            current_date_str = datetime.now().strftime("%Y-%m-%d")
            logger.debug(f"Setting system prompt with current date: {current_date_str}")
            self.system_prompt = DEFAULT_SYSTEM_PROMPT.format(current_date=current_date_str)
            
            self.checkpointer = checkpointer
            logger.debug(f"Checkpointer configured: {type(checkpointer).__name__ if checkpointer else 'None'}")
            
            # Initialize Tavily client
            tavily_api_key = os.getenv("TAVILY_API_KEY")
            if not tavily_api_key:
                logger.error("TAVILY_API_KEY environment variable not set")
                raise ValueError("TAVILY_API_KEY environment variable is required")
            
            logger.debug("Initializing Tavily client")
            self.tavily_client = TavilyClient(api_key=tavily_api_key)
            
            logger.info("SimpleChatbot initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Error initializing SimpleChatbot: {str(e)}", exc_info=True)
            raise

    def router(self, state: State, config: dict):
        logger.info("Router node called")
        logger.debug(f"Router config: {config}")
        logger.debug(f"Router state keys: {list(state.keys())}")
        
        try:
            dispatch_custom_event("router", "routing...", config=config)
            
            logger.debug("Parsing messages for router")
            messages = parse_messages(state)
            logger.debug(f"Parsed messages length: {len(messages)} characters")
            logger.debug(f"Messages preview: {messages[:200]}...")
            
            logger.debug("Invoking router chain")
            router_response = self.router_chain.invoke({"conversation": messages})
            
            route_decision = router_response.content.strip()
            logger.info(f"Router decision: {route_decision}")
            
            if route_decision not in ["tavily", "chatbot"]:
                logger.warning(f"Unexpected router decision: {route_decision}. Defaulting to 'chatbot'")
                route_decision = "chatbot"
            
            return {"router_decision": route_decision}
            
        except Exception as e:
            logger.error(f"Error in router node: {str(e)}", exc_info=True)
            logger.warning("Router error - defaulting to chatbot")
            return {"router_decision": "chatbot"}

    def tavily_node(self, state: State, config: dict):
        logger.info("Tavily node called - performing web search")
        logger.debug(f"Tavily config: {config}")
        logger.debug(f"Tavily state keys: {list(state.keys())}")
        
        try:
            dispatch_custom_event("tavily_status", "searching the web...", config=config)

            logger.debug("Parsing messages for Tavily search (last 30 messages)")
            messages = parse_messages(state, num_messages=30)
            logger.debug(f"Parsed messages for search: {len(messages)} characters")
            logger.debug(f"Search query preview: {messages[:300]}...")

            logger.info("Performing Tavily search with auto parameters")
            search_results = self.tavily_client.search(query=messages, auto_parameters=True)
            
            logger.debug(f"Tavily search completed")
            logger.debug(f"Search results keys: {list(search_results.keys()) if search_results else 'None'}")
            
            if search_results and "results" in search_results:
                logger.info(f"Found {len(search_results['results'])} search results")
                for i, result in enumerate(search_results["results"][:3]):  # Log first 3 results
                    logger.debug(f"Result {i+1}: {result.get('title', 'No title')} - {result.get('url', 'No URL')}")
            else:
                logger.warning("No search results returned from Tavily")

            logger.debug("Converting Tavily results to documents")
            documents = tavily_results_to_documents(search_results)
            logger.debug(f"Created {len(documents)} documents from search results")
            
            logger.debug("Formatting documents for LLM")
            formatted_results = format_documents_for_llm(documents)
            logger.debug(f"Formatted results length: {len(formatted_results)} characters")
            
            # Dispatch custom events
            if search_results and "auto_parameters" in search_results:
                logger.debug(f"Auto parameters: {search_results['auto_parameters']}")
                dispatch_custom_event(
                    "auto_tavily_parameters", search_results["auto_parameters"], config=config
                )
            
            if search_results and "results" in search_results:
                dispatch_custom_event("tavily_results", search_results["results"], config=config)

            logger.debug("Invoking Tavily chain with search results")
            response = self.tavily_chain.invoke(
                {
                    "system_prompt": self.system_prompt,
                    "search_results": formatted_results,
                    "messages": messages,
                }
            )
            
            logger.info("Tavily node completed successfully")
            logger.debug(f"Response type: {type(response)}")
            logger.debug(f"Response content preview: {str(response)[:200]}...")
            
            return {"messages": [response]}
            
        except Exception as e:
            logger.error(f"Error in Tavily node: {str(e)}", exc_info=True)
            # Return an error message instead of crashing
            error_message = f"I encountered an error while searching the web: {str(e)}"
            logger.warning(f"Returning error message to user: {error_message}")
            return {"messages": [{"role": "assistant", "content": error_message}]}

    def chatbot_node(self, state: State, config: dict):
        logger.info("Chatbot node called - generating response from knowledge")
        logger.debug(f"Chatbot config: {config}")
        logger.debug(f"Chatbot state keys: {list(state.keys())}")
        
        try:
            dispatch_custom_event("chatbot_response", "thinking...", config=config)
            
            logger.debug("Parsing messages for chatbot (last 30 messages)")
            messages = parse_messages(state, num_messages=30)
            logger.debug(f"Parsed messages for chatbot: {len(messages)} characters")
            logger.debug(f"Messages preview: {messages[:300]}...")
            
            logger.debug("Invoking chatbot chain")
            response = self.chatbot_chain.invoke(
                {"system_prompt": self.system_prompt, "messages": messages}
            )
            
            logger.info("Chatbot node completed successfully")
            logger.debug(f"Response type: {type(response)}")
            logger.debug(f"Response content preview: {str(response)[:200]}...")
            
            return {"messages": [response]}
            
        except Exception as e:
            logger.error(f"Error in chatbot node: {str(e)}", exc_info=True)
            # Return an error message instead of crashing
            error_message = f"I encountered an error while generating a response: {str(e)}"
            logger.warning(f"Returning error message to user: {error_message}")
            return {"messages": [{"role": "assistant", "content": error_message}]}

    def build_graph(self):
        """Build and compile the graph"""
        logger.info("Building LangGraph workflow")
        
        try:
            logger.debug("Creating StateGraph instance")
            graph_builder = StateGraph(State)

            # Add nodes
            logger.debug("Adding router node")
            graph_builder.add_node("router", self.router)

            logger.debug("Adding tavily node")
            graph_builder.add_node("tavily", self.tavily_node)

            logger.debug("Adding chatbot node")
            graph_builder.add_node("chatbot", self.chatbot_node)

            # Add edges
            logger.debug("Adding edge: START -> router")
            graph_builder.add_edge(START, "router")

            def determine_route(output):
                route = output["router_decision"].strip()
                logger.debug(f"Route determination function called with: {route}")
                return route

            logger.debug("Adding conditional edges from router")
            graph_builder.add_conditional_edges(
                "router",
                determine_route,
                {"tavily": "tavily", "chatbot": "chatbot"},
            )

            logger.debug("Adding edge: tavily -> END")
            graph_builder.add_edge("tavily", END)
            
            logger.debug("Adding edge: chatbot -> END")
            graph_builder.add_edge("chatbot", END)

            logger.debug("Compiling graph with checkpointer")
            compiled_graph = graph_builder.compile(checkpointer=self.checkpointer)
            
            logger.info("Graph compilation completed successfully")
            return compiled_graph
            
        except Exception as e:
            logger.error(f"Error building graph: {str(e)}", exc_info=True)
            raise
