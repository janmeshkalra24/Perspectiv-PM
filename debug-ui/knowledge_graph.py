from typing import Dict, List, Any, Optional, Set
import logging
import os
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import BaseCallbackHandler
from pydantic import BaseModel, Field
from pydantic.config import ConfigDict
from langchain_core.globals import set_verbose, set_debug
from neo4j import GraphDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure LangChain settings
set_verbose(True)

# Import LangChain components
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI

# Configure Pydantic for LangChain
class ChainConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

class BaseGraphStore(ABC):
    """Base class for graph store implementations."""
    
    @abstractmethod
    def query(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """Execute a query against the graph store."""
        pass

    @abstractmethod
    def refresh_schema(self) -> None:
        """Refresh the schema information."""
        pass

    @abstractmethod
    def get_schema(self) -> str:
        """Get the schema as a string."""
        pass

class Neo4jCustomStore(BaseGraphStore):
    """Custom Neo4j graph store implementation."""
    
    def __init__(self, driver):
        """Initialize with Neo4j driver."""
        self.driver = driver
    
    def query(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """Execute a Cypher query."""
        with self.driver.session() as session:
            result = session.run(query, params or {})
            return [dict(record) for record in result]
    
    def refresh_schema(self):
        """Not implemented as we handle schema separately."""
        pass

    def get_schema(self) -> str:
        """Return empty schema as we handle it separately."""
        return ""

class ScreenKnowledgeGraph:
    """Manages a knowledge graph for screen content understanding."""
    
    def __init__(self, model_name: str = 'models/gemini-2.0-flash-lite'):
        """Initialize the knowledge graph.
        
        Args:
            model_name: Name of the Gemini model to use
        """
        self.model_name = model_name
        
        # Initialize Neo4j connection
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        username = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")
        
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.graph_store = Neo4jCustomStore(self.driver)
        
        # Initialize Gemini model for LangChain
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3,
            top_p=1,
            top_k=32,
            max_output_tokens=1024,
        )
        
        # Create graph schema
        self._initialize_schema()
        
        # Cache schema information
        self._refresh_schema_info()
        
        # Initialize the QA chain with custom query function
        self.qa_chain = self._create_qa_chain()
    
    def query(self, cypher_query: str, params: Dict = None) -> List[Dict]:
        """Execute a Cypher query and return results."""
        return self.graph_store.query(cypher_query, params)
    
    def _initialize_schema(self):
        """Initialize the knowledge graph schema."""
        # Create constraints and indexes
        cypher_commands = [
            "CREATE CONSTRAINT frame_id IF NOT EXISTS FOR (f:Frame) REQUIRE f.frame_id IS UNIQUE",
            "CREATE CONSTRAINT element_id IF NOT EXISTS FOR (e:Element) REQUIRE e.element_id IS UNIQUE",
            "CREATE INDEX frame_timestamp IF NOT EXISTS FOR (f:Frame) ON (f.timestamp)",
        ]
        
        for command in cypher_commands:
            try:
                self.query(command)
            except Exception as e:
                logger.error(f"Error creating schema: {e}")
    
    def _create_qa_chain(self):
        """Create a custom QA chain for graph querying."""
        # Get schema information
        schema = self.get_structured_schema()
        
        # Format schema for the prompt
        schema_str = "Node Types and Properties:\n"
        for node_type, props in schema["node_props"].items():
            schema_str += f"- {node_type}: {', '.join(props.keys())}\n"
        
        schema_str += "\nRelationships:\n"
        for rel_type, rel_info in schema["relationships"].items():
            schema_str += f"- {rel_info['start']}-[{rel_type}]->{rel_info['end']}\n"

        # Store schema string for later use
        self.schema_str = schema_str

        # Create Cypher generation chain
        cypher_prompt = PromptTemplate(
            template="""Task: Generate Cypher query to answer the question about the screen recording.
Context: The graph has the following schema:

{graph_schema}

Question: {question}

Generate a Cypher query to find relevant information from the graph.
The query should:
1. Consider temporal relationships between frames
2. Look for relevant UI elements and their relationships
3. Return information that helps answer the question

Cypher query:""",
            input_variables=["question", "graph_schema"]
        )
        
        # Create QA chain
        qa_prompt = PromptTemplate(
            template="""Based on the Cypher query results, answer the following question about the screen recording.
Question: {question}
Cypher query: {query}
Query results: {result}
Answer: Let me analyze the results and provide an answer.""",
            input_variables=["question", "query", "result"]
        )
        
        # Store the prompts and model for later use
        self.cypher_prompt = cypher_prompt
        self.qa_prompt = qa_prompt
        
        return self  # Return self instead of a chain instance

    async def ainvoke(self, inputs):
        """Process inputs to generate a question answer from the knowledge graph.
        
        This replaces the GraphCypherQAChain functionality with our custom implementation.
        """
        question = inputs.get("question", "")
        
        # 1. Generate Cypher query
        cypher_inputs = {
            "question": question,
            "graph_schema": self.schema_str
        }
        cypher_response = self.llm.invoke(self.cypher_prompt.format(**cypher_inputs))
        cypher_query = cypher_response.text.strip()
        
        # 2. Execute the Cypher query
        try:
            result = self.query(cypher_query)
        except Exception as e:
            logger.error(f"Error executing Cypher query: {e}")
            result = [{"error": str(e)}]
            
        # 3. Generate answer based on query results
        qa_inputs = {
            "question": question,
            "query": cypher_query,
            "result": str(result)
        }
        qa_response = self.llm.invoke(self.qa_prompt.format(**qa_inputs))
        answer = qa_response.text.strip()
        
        # Return formatted result with intermediate steps
        return {
            "result": answer,
            "intermediate_steps": {
                "query": cypher_query,
                "result": result
            }
        }

    def __del__(self):
        """Cleanup Neo4j connection."""
        if hasattr(self, 'driver'):
            self.driver.close()

    def add_frame_to_graph(self, frame_data: Dict[str, Any]):
        """Add a frame and its elements to the knowledge graph.
        
        Args:
            frame_data: Dictionary containing frame information
        """
        frame_id = frame_data.get("frame_index")
        timestamp = frame_data.get("metadata", {}).get("timestamp", 0)
        description = frame_data.get("description", "")
        
        # Create frame node with more descriptive label
        frame_query = """
        MERGE (f:Frame {frame_id: $frame_id})
        SET f.timestamp = $timestamp,
            f.description = $description,
            f.label = $label
        RETURN f
        """
        
        frame_label = f"Frame {frame_id} at {timestamp:.1f}s"
        self.query(frame_query, {
            "frame_id": frame_id,
            "timestamp": timestamp,
            "description": description,
            "label": frame_label
        })
        
        # Extract UI elements using LLM
        prompt = f"""Analyze this screen description and extract UI elements.
For each element, identify:
1. Type (window, app, button, text, link, etc.)
2. Content (what it contains/represents)
3. Technical terms or jargon
4. User mentions or references
5. Tasks or todos
6. Warnings or errors

Description: {description}

Return as a list of dictionaries:
[{{"type": "window", "content": "Chrome Browser", "category": "application"}}, ...]"""
        
        try:
            response = self.llm.invoke(prompt)
            elements = eval(response.text)
            
            # Add each element and create relationships
            for element in elements:
                element_type = element.get("type", "unknown")
                content = element.get("content", "")
                category = element.get("category", "")
                
                # Create unique ID for element
                element_id = f"{frame_id}_{element_type}_{hash(content)}"
                
                # Add element with meaningful label and content
                element_query = """
                MATCH (f:Frame {frame_id: $frame_id})
                MERGE (e:Element {element_id: $element_id})
                SET e.type = $type,
                    e.content = $content,
                    e.category = $category,
                    e.label = $label
                MERGE (f)-[:CONTAINS]->(e)
                """
                
                # Create meaningful label based on type and content
                element_label = f"{element_type}: {content[:30]}"
                
                self.query(element_query, {
                    "frame_id": frame_id,
                    "element_id": element_id,
                    "type": element_type,
                    "content": content,
                    "category": category,
                    "label": element_label
                })
                
                # Create relationships between related elements
                if len(elements) > 1:
                    related_query = """
                    MATCH (e1:Element {element_id: $element_id})
                    MATCH (e2:Element)
                    WHERE e2.frame_id = $frame_id 
                    AND e2.element_id <> $element_id
                    AND (
                        e1.type = e2.type OR
                        e1.category = e2.category OR
                        e1.content CONTAINS e2.content OR
                        e2.content CONTAINS e1.content
                    )
                    MERGE (e1)-[:RELATED_TO]->(e2)
                    """
                    self.query(related_query, {
                        "frame_id": frame_id,
                        "element_id": element_id
                    })
        except Exception as e:
            logger.error(f"Error extracting UI elements: {e}")
            return []
    
    async def query_knowledge_graph(self, question: str, context: Dict[str, Any]) -> str:
        """Query the knowledge graph to answer questions.
        
        Args:
            question: User's question
            context: Current context information
            
        Returns:
            Answer based on the knowledge graph
        """
        try:
            # Add current frame context if available
            if context.get("currentFrame"):
                self.add_frame_to_graph(context["currentFrame"])
            
            # Add historical context
            for frame in context.get("history", []):
                self.add_frame_to_graph(frame)
            
            # Query the graph using our custom implementation
            result = await self.ainvoke({
                "question": question,
                "context": context
            })
            
            return result["result"]
            
        except Exception as e:
            logger.error(f"Error querying knowledge graph: {e}")
            return "I encountered an error while searching the knowledge graph. Falling back to direct question answering."

    def _refresh_schema_info(self):
        """Refresh schema information from the database."""
        # Get node labels
        node_labels_query = """
        CALL db.labels() YIELD label
        RETURN collect(label) as labels
        """
        result = self.query(node_labels_query)
        self.node_labels = result[0].get('labels', []) if result else []

        # Get relationship types
        rel_types_query = """
        CALL db.relationshipTypes() YIELD relationshipType
        RETURN collect(relationshipType) as types
        """
        result = self.query(rel_types_query)
        self.relationship_types = result[0].get('types', []) if result else []

        # Get node properties
        self.node_properties = {}
        for label in self.node_labels:
            props_query = f"""
            MATCH (n:{label})
            RETURN keys(n) as props
            LIMIT 1
            """
            result = self.query(props_query)
            if result:
                self.node_properties[label] = result[0].get('props', [])

    def get_structured_schema(self, 
                            include_types: Optional[Set[str]] = None,
                            exclude_types: Optional[Set[str]] = None) -> Dict:
        """Get structured schema information for the graph.
        
        Args:
            include_types: Optional set of node types to include
            exclude_types: Optional set of node types to exclude
            
        Returns:
            Dictionary containing schema information in the format:
            {
                "node_props": {node_label: {prop_name: prop_type}},
                "relationships": {rel_type: {"start": start_type, "end": end_type}}
            }
        """
        # Filter node labels based on include/exclude
        node_labels = self.node_labels
        if include_types:
            node_labels = [l for l in node_labels if l in include_types]
        if exclude_types:
            node_labels = [l for l in node_labels if l not in exclude_types]
        
        # Build node properties dictionary
        node_props = {}
        for label in node_labels:
            props = self.node_properties.get(label, [])
            node_props[label] = {prop: "Any" for prop in props}
        
        # Build relationships dictionary
        # Query to get relationship type information
        rel_query = """
        MATCH (start)-[r]->(end)
        RETURN DISTINCT type(r) as rel_type, 
               labels(start)[0] as start_label, 
               labels(end)[0] as end_label
        """
        relationships = {}
        try:
            rel_results = self.query(rel_query)
            for result in rel_results:
                rel_type = result.get('rel_type')
                if rel_type:
                    relationships[rel_type] = {
                        "start": result.get('start_label', 'Node'),
                        "end": result.get('end_label', 'Node')
                    }
        except Exception as e:
            logger.error(f"Error getting relationship information: {e}")
            # Provide default relationship structure if query fails
            relationships = {
                "CONTAINS": {"start": "Frame", "end": "Element"},
                "RELATED_TO": {"start": "Element", "end": "Element"}
            }
        
        return {
            "node_props": node_props,
            "relationships": relationships
        } 