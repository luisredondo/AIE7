"""D&D specialist node with Qdrant RAG integration."""

import json
from typing import List, Dict, Any, Optional

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from ollama_deep_researcher.configuration import Configuration
from ollama_deep_researcher.state import SummaryState
from ollama_deep_researcher.utils import strip_thinking_tokens


class QdrantRAG:
    """RAG system using Qdrant vector database for D&D documents."""
    
    def __init__(
        self, 
        host: str = "127.0.0.1", 
        port: int = 6334,
        collection_name: str = "DnD_Documents",
        api_key: Optional[str] = None
    ):
        """Initialize Qdrant RAG system.
        
        Args:
            host: Qdrant server host
            port: Qdrant server port
            collection_name: Name of the collection containing D&D documents
            api_key: Optional API key for Qdrant
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.api_key = api_key
        self._client = None
    
    def _get_client(self):
        """Get or create Qdrant client."""
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
                self._client = QdrantClient(
                    host=self.host,
                    port=self.port,
                    api_key=self.api_key
                )
            except ImportError:
                raise ImportError(
                    "Qdrant client not installed. Please install it with: "
                    "pip install qdrant-client"
                )
        return self._client
    
    def search(
        self, 
        query: str, 
        limit: int = 5,
        score_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Search for relevant D&D documents.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of relevant documents with their content and metadata
        """
        client = self._get_client()
        
        try:
            from langchain_ollama import OllamaEmbeddings
            
            # Initialize embeddings model
            embeddings = OllamaEmbeddings(
                model="mxbai-embed-large",
                base_url="http://localhost:11434"
            )
            
            # Generate query embedding
            query_vector = embeddings.embed_query(query)
            
            # Search in Qdrant
            search_results = client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold
            )
            
            # Format results
            results = []
            for hit in search_results:
                results.append({
                    "content": hit.payload.get("content", ""),
                    "metadata": hit.payload.get("metadata", {}),
                    "score": hit.score
                })
            
            return results
            
        except Exception as e:
            print(f"Error searching Qdrant: {e}")
            return []


def dnd_specialist_node(state: SummaryState, config: RunnableConfig):
    """LangGraph node that provides D&D expertise using RAG with Qdrant.
    
    This node searches for relevant D&D information in the Qdrant vector database
    and uses an LLM to provide expert analysis and answers about D&D topics.
    
    Args:
        state: Current graph state containing the research topic and context
        config: Configuration for the runnable, including LLM provider settings
        
    Returns:
        Dictionary with state update, including D&D-specific insights
    """
    configurable = Configuration.from_runnable_config(config)
    
    # Initialize Qdrant RAG
    rag = QdrantRAG(
        host="127.0.0.1",
        port=6334,
        collection_name="DnD_Documents"
    )
    
    # Extract D&D-related query from the research topic
    research_topic = state.research_topic
    
    # Search for relevant D&D documents
    dnd_documents = rag.search(research_topic, limit=5)
    
    # Format the retrieved documents
    context_parts = []
    for doc in dnd_documents:
        context_parts.append(f"Document (score: {doc['score']:.2f}):\n{doc['content']}")
    
    dnd_context = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant D&D documents found."
    
    # Prepare the prompt for the LLM
    system_prompt = """You are a Dungeons & Dragons expert assistant with deep knowledge of:
- D&D 5th Edition rules, mechanics, and lore
- Character creation and optimization
- Monster stats and behaviors
- Campaign settings and world-building
- Adventure modules and storytelling
- Magic items, spells, and abilities

Use the provided context from the D&D document collection to provide accurate and helpful information.
If the context doesn't contain relevant information, use your general D&D knowledge.
Always cite specific rules, page numbers, or sources when available."""
    
    human_prompt = f"""Research Topic: {research_topic}

Retrieved D&D Context:
{dnd_context}

Based on the above context and your D&D expertise, provide a comprehensive response about the topic.
Include specific rules references, mechanics explanations, and practical advice where relevant."""
    
    # Get LLM based on configuration
    if configurable.llm_provider == "lmstudio":
        from ollama_deep_researcher.lmstudio import ChatLMStudio
        llm = ChatLMStudio(
            base_url=configurable.lmstudio_base_url,
            model=configurable.local_llm,
            temperature=0.3,
        )
    else:  # Default to Ollama
        from langchain_ollama import ChatOllama
        llm = ChatOllama(
            base_url=configurable.ollama_base_url,
            model=configurable.local_llm,
            temperature=0.3,
        )
    
    # Generate response
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]
    
    result = llm.invoke(messages)
    
    # Process the response
    dnd_insights = result.content
    if configurable.strip_thinking_tokens:
        dnd_insights = strip_thinking_tokens(dnd_insights)
    
    # Create comprehensive D&D analysis as the primary summary
    updated_summary = f"""# Research Report: {state.research_topic}

## D&D Specialist Analysis

{dnd_insights}

### Sources Consulted
{f"- Retrieved {len(dnd_documents)} relevant documents from D&D knowledge base" if dnd_documents else "- No specific D&D documents were found, using general D&D knowledge"}
- Qdrant Collection: DnD_Documents
- Embedding Model: mxbai-embed-large"""
    
    # Format sources for display
    sources_list = []
    if dnd_documents:
        for i, doc in enumerate(dnd_documents, 1):
            sources_list.append(f"[{i}] D&D Document (relevance: {doc['score']:.2%})")
    
    return {
        "running_summary": updated_summary,
        "web_research_results": [f"D&D RAG Results:\n{dnd_context}"],
        "sources_gathered": sources_list if sources_list else ["D&D Knowledge Base (Qdrant)"],
        "research_loop_count": 1  # Mark as having completed one research iteration
    }


def should_consult_dnd_specialist(state: SummaryState, config: RunnableConfig) -> bool:
    """Determine if the D&D specialist should be consulted.
    
    Args:
        state: Current graph state
        config: Configuration for the runnable
        
    Returns:
        True if the topic appears to be D&D-related
    """
    research_topic = state.research_topic.lower() if state.research_topic else ""
    search_query = state.search_query.lower() if hasattr(state, 'search_query') and state.search_query else ""
    
    # Combine topic and query for better detection
    combined_text = f"{research_topic} {search_query}"
    
    # D&D-specific terms (high confidence)
    dnd_specific = [
        "d&d", "dnd", "dungeons and dragons", "dungeons & dragons",
        "5e", "5th edition", "3.5e", "3rd edition", "pathfinder",
        "dungeon master", "dm guide", "dmg", "player's handbook", "phb",
        "monster manual", "mm", "xanathar", "tasha", "volo", "mordenkainen",
        "forgotten realms", "eberron", "ravnica", "strahd", "barovia"
    ]
    
    # D&D-related gameplay terms (medium confidence)
    dnd_gameplay = [
        "armor class", "hit points", "hp", "ac", "saving throw",
        "death save", "ability check", "skill check", "proficiency bonus",
        "advantage", "disadvantage", "inspiration", "initiative",
        "action economy", "bonus action", "reaction", "concentration",
        "spell slot", "cantrip", "ritual", "components",
        "multiclassing", "multiclass", "feat", "ability score",
        "challenge rating", "cr", "experience points", "xp"
    ]
    
    # Check for specific terms first (high confidence)
    if any(term in combined_text for term in dnd_specific):
        return True
    
    # Check for gameplay terms (need at least 2 matches for confidence)
    gameplay_matches = sum(1 for term in dnd_gameplay if term in combined_text)
    if gameplay_matches >= 2:
        return True
    
    # Check for D&D classes and races together
    classes = ["barbarian", "bard", "cleric", "druid", "fighter", 
               "monk", "paladin", "ranger", "rogue", "sorcerer", 
               "warlock", "wizard", "artificer"]
    races = ["dwarf", "elf", "halfling", "human", "dragonborn", 
             "gnome", "tiefling", "orc", "goliath"]
    
    has_class = any(cls in combined_text for cls in classes)
    has_race = any(race in combined_text for race in races)
    
    # If mentions both a class and race, likely D&D
    if has_class and has_race:
        return True
    
    # If mentions class/race with other D&D context words
    if (has_class or has_race) and any(
        word in combined_text for word in 
        ["character", "campaign", "adventure", "quest", "party", "spell", "magic"]
    ):
        return True
    
    return False