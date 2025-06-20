import logging
from langchain_core.messages import AIMessage, HumanMessage

logger = logging.getLogger(__name__)


def parse_messages(state, num_messages=None):
    logger.debug(f"parse_messages called with num_messages={num_messages}")
    
    # Check if messages exist and are in a list
    if "messages" not in state or not state["messages"]:
        logger.warning("No messages found in state")
        return ""

    messages = state["messages"]
    logger.debug(f"Found {len(messages)} total messages in state")

    # If num_messages is specified, get only the most recent n messages
    if num_messages is not None and num_messages > 0:
        original_count = len(messages)
        # Simply take the n most recent messages
        messages = messages[-min(num_messages, len(messages)) :]
        logger.debug(f"Limited to {len(messages)} most recent messages (from {original_count})")

    # Find the last human message index (simplest approach)
    last_human_index = -1
    for i in range(len(messages) - 1, -1, -1):  # Search backwards
        msg = messages[i]
        if (
            isinstance(msg, HumanMessage)
            or (hasattr(msg, "type") and msg.type == "Human")
            or (
                hasattr(msg, "content")
                and msg.content.lower().strip().startswith("human:")
            )
        ):
            last_human_index = i
            logger.debug(f"Found last human message at index {i}")
            break
    
    if last_human_index == -1:
        logger.warning("No human message found in the conversation")

    # Process all messages
    formatted_messages = []
    for i, msg in enumerate(messages):
        # Get role and content
        if hasattr(msg, "type"):
            role = msg.type
        elif isinstance(msg, HumanMessage):
            role = "Human"
        elif isinstance(msg, AIMessage):
            role = "AI"
        else:
            role = "Unknown"
            logger.debug(f"Unknown message type at index {i}: {type(msg)}")

        content = msg.content if hasattr(msg, "content") else str(msg)
        logger.debug(f"Message {i}: {role} - {len(content)} characters")

        # Format the message
        if i == last_human_index:
            formatted_messages.append(
                f"[MOST RECENT HUMAN MESSAGE] \n {role}: {content}"
            )
        else:
            formatted_messages.append(f"{role}: {content}")

    result = "\n".join(formatted_messages)
    logger.debug(f"parse_messages returning {len(result)} characters")
    return result


def tavily_results_to_documents(tavily_response):
    """
    Convert Tavily search results to a list of LangChain Document objects.

    Args:
        tavily_response (dict): The response from a Tavily search query

    Returns:
        List[Document]: A list of LangChain Document objects with content and metadata
    """
    from langchain.schema import Document

    documents = []

    if not tavily_response or "results" not in tavily_response:
        return documents

    for result in tavily_response["results"]:
        # Create a Document with content from the result
        doc = Document(
            page_content=result.get("content", ""),
            metadata={
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "score": result.get("score", 0.0),
            },
        )
        documents.append(doc)

    return documents


def format_documents_for_llm(documents):
    """
    Format a list of Document objects into a well-structured string for language models.

    Args:
        documents (List[Document]): A list of LangChain Document objects with content and metadata

    Returns:
        str: A formatted string with search results organized for easy consumption by LLMs
    """
    if not documents:
        return "No search results found."

    formatted_results = []

    for i, doc in enumerate(documents, 1):
        # Extract metadata
        title = doc.metadata.get("title", "No title")
        url = doc.metadata.get("url", "No URL")

        # Format the document content
        content = doc.page_content.strip()

        # Create a formatted entry
        formatted_doc = f"\nRESULT {i}:\n"
        formatted_doc += f"Title: {title}\n"
        formatted_doc += f"URL: {url}\n"
        formatted_doc += f"Content: {content}\n"

        formatted_results.append(formatted_doc)

    # Join all formatted results with a separator
    return "\n" + "-" * 40 + "\n".join(formatted_results)
