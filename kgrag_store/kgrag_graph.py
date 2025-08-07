import os
import uuid
import json
from typing import (
    Literal,
    LiteralString,
    AsyncGenerator,
    Any,
    Optional
)
from neo4j import GraphDatabase, Driver
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.documents import Document

from .kgrag_ollama import ollama_pull
from log import logger, get_metadata
from .kgrag_components import GraphComponents
from .kgrag_prompt import AGENT_PROMPT, parser_prompt
from .kgrag_vector import KGragVectorStore

TypeModel = Literal["openai", "ollama", "vllm"]


class KGragGraph(KGragVectorStore):
    """
    MemoryGraph is a class that manages a memory graph using
    Qdrant as the vector store.
    It provides methods to add messages, delete collections,
    and retrieve data.
    """

    neo4j_url: str
    neo4j_username: str
    neo4j_password: str
    neo4j_database: str | None = None
    neo4j_driver: Driver | None = None
    llm_type: TypeModel = "openai"
    llm_model: str = "gpt-4o-2024-08-06"
    llm_model_url: str | None = None
    temperature: float = 0.0
    client_llm: ChatOpenAI | ChatOllama

    def __init__(self, **kwargs):
        """
        Initialize the MemoryStoreGraph with the provided parameters.
        Args:
            **kwargs: Additional keyword arguments for configuration.
        Raises:
            ValueError: If the Neo4j URL, username, or password
            is not provided.
        """
        super().__init__(**kwargs)
        self.neo4j_url = kwargs.get("neo4j_url", os.getenv("NEO4J_URL"))
        self.neo4j_username = kwargs.get(
            "neo4j_username", os.getenv("NEO4J_USERNAME")
        )
        self.neo4j_password = kwargs.get(
            "neo4j_password", os.getenv("NEO4J_PASSWORD")
        )
        self.neo4j_database = kwargs.get("neo4j_database", None)
        self.temperature = kwargs.get("temperature", 0.0)
        self.llm_type = kwargs.get("llm_type", os.getenv("LLM_TYPE", "openai"))
        if self.llm_type not in ["openai", "ollama", "vllm"]:
            msg: str = (
                f"Invalid llm_type: {self.llm_type}. "
                "Must be one of 'openai', 'ollama', or 'vllm'."
            )
            logger.error(msg)
            raise ValueError(msg)

        if self.llm_type == "ollama":
            self.llm_model = kwargs.get(
                "llm_model",
                os.getenv("LLM_MODEL", "llama3.2")
            )
            self.llm_model_url = kwargs.get(
                "llm_model_url",
                os.getenv("LLM_URL", "http://localhost:11434")
            )
            if self.llm_model_url is None:
                logger.warning(
                    "llm_model_url not set, using default Ollama base URL",
                    extra=get_metadata(thread_id=str(self.thread))
                )
                raise ValueError("llm_model_url must be set")
            logger.debug(
                (
                    f"Pulling Ollama model {self.llm_model} "
                    f"from {self.llm_model_url}"
                ),
                extra=get_metadata(thread_id=str(self.thread))
            )
            ollama_pull(
                ollama_url=self.llm_model_url,
                model_name=self.llm_model
            )
        elif self.llm_type == "vllm":
            self.llm_model = kwargs.get(
                "llm_model",
                os.getenv("LLM_MODEL", "gpt-4o-2024-08-06")
            )
            self.llm_model_url = kwargs.get(
                "llm_model_url",
                os.getenv("LLM_URL", "http://localhost:10500")
            )
            if self.llm_model_url is None:
                logger.warning(
                    "llm_model_url not set, using default VLLM base URL",
                    extra=get_metadata(thread_id=str(self.thread))
                )
                raise ValueError("llm_model_url must be set")
        else:
            self.llm_model = kwargs.get(
                "llm_model",
                os.getenv("LLM_MODEL", "gpt-4o-2024-08-06")
            )
            self.llm_model_url = kwargs.get("llm_model_url", None)

        if not self.neo4j_url:
            msg: str = (
                "Neo4j URL not provided. Please set the 'neo4j_url' parameter "
                "or the 'NEO4J_URL' environment variable."
            )
            logger.warning(msg)

        if not self.neo4j_username:
            msg: str = (
                "Neo4j username not provided. "
                " Please set the 'neo4j_username' parameter "
                "or the 'NEO4J_USERNAME' environment variable."
            )
            logger.warning(msg)

        if not self.neo4j_password:
            msg: str = (
                "Neo4j password not provided. "
                "Please set the 'neo4j_password' parameter "
                "or the 'NEO4J_PASSWORD' environment variable."
            )
            logger.warning(msg)

        # Initialize Neo4j driver
        if all([self.neo4j_url, self.neo4j_username, self.neo4j_password]):
            self.neo4j_driver = GraphDatabase.driver(
                self.neo4j_url,
                auth=(self.neo4j_username, self.neo4j_password)
            )

            if self.neo4j_database:
                logger.debug(
                    f"Using Neo4j database: {self.neo4j_database}",
                    extra=get_metadata(thread_id=str(self.thread))
                )
                self.create_database_if_not_exists(self.neo4j_database)

        self.client_llm = self._get_model(llm_type=self.llm_type,
                                          llm_model=self.llm_model,
                                          llm_model_url=self.llm_model_url)

    def delete_all_relationships(self):
        """
        Delete all relationships in the Neo4j database.
        This method is useful for clearing the graph before a new ingestion.
        Raises:
            ValueError: If the Neo4j driver is not initialized.
        """

        if not self.neo4j_driver:
            msg: str = "Neo4j driver is not initialized."
            logger.error(msg, extra=get_metadata(thread_id=str(self.thread)))
            raise ValueError(msg)

        with self.neo4j_driver.session() as session:
            session.run("MATCH ()-[r]-() DELETE r")

    def create_database_if_not_exists(self, db_name):
        """
        Create a Neo4j database if it does not already exist.
        Args:
            db_name (str): The name of the database to create.
        Returns:
            None
        Raises:
            ValueError: If the Neo4j driver is not initialized.
        """
        if not self.neo4j_driver:
            msg: str = "Neo4j driver is not initialized."
            logger.error(msg, extra=get_metadata(thread_id=str(self.thread)))
            raise ValueError(msg)

        with self.neo4j_driver.session(database="system") as session:
            # Controlla se il database esiste già
            result = session.run("SHOW DATABASES")
            dbs = [record["name"] for record in result]
            if db_name in dbs:
                logger.debug(
                    f"Il database '{db_name}' esiste già. "
                    "Nessuna azione eseguita."
                )
            else:
                query: LiteralString = "CREATE DATABASE {db_name}"
                session.run(query)
                logger.debug(f"Database '{db_name}' creato.")

    def _ensure_str(self, val) -> str:
        """
        Ensure that the value is a string. If it is a list,
        join the elements into a string.
        Args:
            val: The value to ensure is a string.
        Returns:
            str: The value as a string.
        """
        if isinstance(val, list):
            return ", ".join(str(x) for x in val)
        elif not isinstance(val, str):
            return str(val)
        return val

    def _get_model(self, **kwargs):
        """
        Get the language model to use for generating text.
        Args:
            **kwargs: Additional keyword arguments for model configuration.
            - llm_type (str): The type of language model
                to use (e.g., "openai", "ollama", "vllm").
            - llm_model (str): The name of the language model to use.
            - llm_model_url (str): The URL of the language model if applicable.

        Returns:
            Any: The language model to use.
        Raises:
            ValueError: If the model_embedding_type or
                model_embedding_name is not set.
            Exception: If there is an error during
                the loading of the language model.
        """
        try:

            llm_type = kwargs.get("llm_type", self.llm_type)
            llm_model = kwargs.get("llm_model", self.llm_model)
            llm_model_url = kwargs.get("llm_model_url", self.llm_model_url)

            if llm_type is None:
                logger.error(
                    "llm_type not set, using default type",
                    extra=get_metadata(thread_id=str(self.thread))
                )
                raise ValueError("llm_type must be set")

            if llm_model is None:
                logger.error("llm_model not set, using default model",
                             extra=get_metadata(
                                thread_id=str(self.thread)
                             ))
                raise ValueError("llm_model must be set")
            if (
                (llm_type == 'ollama' or llm_type == 'vllm')
                and llm_model_url is None
            ):
                logger.error(
                    "llm_model_url not set, using default base URL",
                    extra=get_metadata(thread_id=str(self.thread))
                )
                raise ValueError("llm_model_url must be set")

            if llm_type == 'openai':
                logger.info("Using OpenAI LLM",
                            extra=get_metadata(
                                thread_id=str(self.thread)
                            ))
                return ChatOpenAI(model=llm_model,
                                  temperature=self.temperature)
            elif llm_type == 'ollama':
                logger.info(
                    "Using Ollama LLM",
                    extra=get_metadata(thread_id=str(self.thread))
                )
                return ChatOllama(
                    model=llm_model,
                    base_url=llm_model_url,
                    temperature=self.temperature
                )
            elif llm_type == 'vllm':
                logger.info(
                    "Using VLLM",
                    extra=get_metadata(thread_id=str(self.thread))
                )
                return ChatOpenAI(
                    model=llm_model,
                    base_url=llm_model_url,
                    temperature=self.temperature
                )
        except Exception as e:
            msg: str = (
                f"Error during language model loading: {e}"
            )
            logger.error(
                msg,
                extra=get_metadata(thread_id=str(self.thread))
            )
            raise e

    async def llm_parser(
        self,
        prompt_text: str,
        prompt_user: Optional[str] = None
    ) -> GraphComponents:
        """
        Uses OpenAI's LLM to parse the prompt and extract graph components.
        Args:
            prompt (str): The input text containing nodes and relationships.
        Returns:
            GraphComponents: A Pydantic model containing
            the extracted graph components.
        Raises:
            ValueError: If the OpenAI response content is None.
        """
        try:
            prompt_parser: str = parser_prompt(prompt_user)

            prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    prompt_parser
                ),
                ("human", "{input_text}")
            ])

            chain = prompt | self.client_llm
            response = await chain.ainvoke({"input_text": prompt_text})

            if response is None:
                logger.error(
                    "OpenAI response content is None",
                    extra=get_metadata(thread_id=str(self.thread))
                )
                raise ValueError("OpenAI response content is None")

            raw_content = response.content

            logger.debug(f"Raw content from LLM: {raw_content}")

            # Ensure raw_content is a JSON string
            if not isinstance(raw_content, (str, bytes, bytearray)):
                raw_content = json.dumps(raw_content)

            return GraphComponents.model_validate_json(raw_content)
        except Exception as e:
            logger.error(
                f"Error during LLM parsing: {str(e)}",
                extra=get_metadata(thread_id=str(self.thread))
            )
            raise e

    async def extract_graph_components(
        self,
        raw_data: str,
        **kwargs
    ) -> tuple[dict[str, str], list[dict[str, str]]]:
        """
        Extract nodes and relationships from the provided
        raw data using LLM.
        Args:
            raw_data (str): The input text containing nodes and relationships.
            prompt_user (str | None): The user prompt to guide the extraction.
        Returns:
            tuple: A tuple containing a dictionary of nodes and a
                list of relationships.
        """

        prompt_user: str | None = kwargs.get("prompt_user", None)

        prompt: str = (
            f"Extract nodes and relationships from the following text:\n"
            f"{raw_data}"
        )

        logger.debug(
            f"Extracting graph components from raw data: {prompt}",
            extra=get_metadata(thread_id=str(self.thread))
        )
        # Assuming this returns a list of dictionaries
        parsed_response = await self.llm_parser(
            prompt,
            prompt_user=prompt_user
        )
        if not parsed_response:
            msg: str = (
                "Parsed response is empty or does not contain "
                "'graph' attribute"
            )
            logger.error(
                msg,
                extra=get_metadata(thread_id=str(self.thread))
            )
            raise ValueError(msg)

        # Assuming the 'graph' structure is a key in the parsed response
        parsed_response = parsed_response.graph
        nodes = {}
        relationships = []

        for entry in parsed_response:
            target_node = entry.target_node  # Get target_node from the entry
            node = entry.node
            relationship = entry.relationship  # Get relationship if available

            # Add nodes to the dictionary with a unique ID
            if node not in nodes:
                nodes[node] = str(uuid.uuid4())

            if target_node and target_node not in nodes:
                nodes[target_node] = str(uuid.uuid4())

            # Add relationship to the relationships list with node IDs
            if target_node and relationship:
                relationships.append({
                    "source": nodes[node],
                    "target": nodes[target_node],
                    "type": relationship
                })

        msg: str = (
            f"Extracted {len(nodes)} nodes and "
            f"{len(relationships)} relationships from the raw data."
        )
        logger.debug(
            msg,
            extra=get_metadata(
                thread_id=str(self.thread)
            )
        )
        return nodes, relationships

    def ingest_to_neo4j(
        self,
        nodes: dict[str, str],
        relationships: list[dict[str, str]]
    ):
        """
        Ingest nodes and relationships into Neo4j.
        Args:
            nodes (dict): A dictionary of nodes with their
                names and unique IDs.
            relationships (list): A list of relationships, each represented
                as a dictionary with source, target, and type.
        Returns:
            dict: A dictionary of nodes with their names and unique IDs.
        Raises:
            ValueError: If Neo4j driver is not initialized or if nodes
                or relationships are empty.
        """
        try:
            if not self.neo4j_driver:
                msg: str = "Neo4j driver is not initialized."
                logger.error(
                    msg,
                    extra=get_metadata(thread_id=str(self.thread))
                )
                raise ValueError(msg)

            with self.neo4j_driver.session() as session:
                # Create nodes in Neo4j
                for name, node_id in nodes.items():
                    session.run(
                        "CREATE (n:Entity {id: $id, name: $name})",
                        id=node_id,
                        name=name
                    )

                # Create relationships in Neo4j
                for relationship in relationships:
                    session.run(
                        (
                            "MATCH (a:Entity {id: $source_id}), "
                            "(b:Entity {id: $target_id}) "
                            "CREATE (a)-[:RELATIONSHIP {type: $type}]->(b)"
                        ),
                        source_id=relationship["source"],
                        target_id=relationship["target"],
                        type=relationship["type"]
                    )

            return nodes
        except Exception as e:
            logger.error(
                f"Error during Neo4j ingestion: {str(e)}",
                extra=get_metadata(thread_id=str(self.thread))
            )
            raise e

    def _fetch_related_graph(self, entity_ids):
        """
        Fetch a subgraph related to the given entity IDs from Neo4j.
        Args:
            entity_ids (list): A list of entity IDs to fetch related
            nodes and relationships.
        Returns:
            list: A list of dictionaries representing the subgraph,
                where each dictionary contains:
                - "entity": The entity node.
                - "relationship": The relationship to the related node.
                - "related_node": The related node.
        Raises:
            ValueError: If the Neo4j client is not initialized
                or if entity_ids is empty.
        """

        if not self.neo4j_driver:
            msg: str = "Neo4j driver is not initialized."
            logger.error(msg, extra=get_metadata(thread_id=str(self.thread)))
            raise ValueError(msg)

        query = """
        MATCH (e:Entity)-[r1]-(n1)-[r2]-(n2)
        WHERE e.id IN $entity_ids
        RETURN e, r1 as r, n1 as related, r2, n2
        UNION
        MATCH (e:Entity)-[r]-(related)
        WHERE e.id IN $entity_ids
        RETURN e, r, related, null as r2, null as n2
        """
        with self.neo4j_driver.session() as session:
            result = session.run(query, entity_ids=entity_ids)
            subgraph = []
            for record in result:
                subgraph.append({
                    "entity": record["e"],
                    "relationship": record["r"],
                    "related_node": record["related"]
                })
                if record["r2"] and record["n2"]:
                    subgraph.append({
                        "entity": record["related"],
                        "relationship": record["r2"],
                        "related_node": record["n2"]
                    })
        return subgraph

    def _format_graph_context(self, subgraph):
        """
        Format the subgraph into a context suitable for LLM processing.
        Args:
            subgraph (list): A list of dictionaries representing
                the subgraph, where each dictionary contains:
                - "entity": The entity node.
                - "relationship": The relationship to the related node.
                - "related_node": The related node.
        Returns:
            dict: A dictionary containing:
                - "nodes": A list of unique node names.
                - "edges": A list of edges in the format "entity
                    relationship related_node".
        Raises:
            ValueError: If the subgraph is empty or if
                the Neo4j driver is not initialized.
        """

        nodes = set()
        edges = []

        for entry in subgraph:
            entity = entry["entity"]
            related = entry["related_node"]
            relationship = entry["relationship"]

            nodes.add(entity["name"])
            nodes.add(related["name"])

            edges.append(
                f"{entity['name']} {relationship['type']} {related['name']}"
            )

        return {"nodes": list(nodes), "edges": edges}

    async def _stream(self, graph_context, user_query):
        """
        Run the GraphRAG process using the provided
            graph context and user query.
        Args:
            graph_context (dict): A dictionary containing the graph
                context with nodes and edges.
            user_query (str): The user's query to be answered
                using the graph context.
            **kwargs: Additional keyword arguments for LLM configuration.
        Returns:
            str: The response from the LLM based on the graph
                context and user query.
        Raises:
            ValueError: If the graph context is not provided or
                if the user query is empty.
            Exception: If there is an error querying the LLM.
        """
        try:
            input, chain = self._get_chain_graph(user_query, graph_context)

            async for response in chain.astream(input):
                if response is None:
                    logger.error(
                        "OpenAI response content is None",
                        extra=get_metadata(thread_id=str(self.thread))
                    )
                    raise ValueError("OpenAI response content is None")

                if isinstance(response.content, list):
                    answer_text = "\n".join(
                        str(item) for item in response.content
                    )
                else:
                    answer_text = str(response.content)

                yield answer_text.strip()

        except Exception as e:
            logger.error(
                f"Error during LLM processing: {str(e)}",
                extra=get_metadata(thread_id=str(self.thread))
            )
            raise e

    async def _run(self, graph_context, user_query):
        """
        Run the GraphRAG process using the provided
            graph context and user query.
        Args:
            graph_context (dict): A dictionary containing the graph
                context with nodes and edges.
            user_query (str): The user's query to be answered
                using the graph context.
            **kwargs: Additional keyword arguments for LLM configuration.
        Returns:
            str: The response from the LLM based on the graph
                context and user query.
        Raises:
            ValueError: If the graph context is not provided or
                if the user query is empty.
            Exception: If there is an error querying the LLM.
        """
        try:
            input, chain = self._get_chain_graph(user_query, graph_context)

            response = await chain.ainvoke(input)

            if response is None:
                logger.error(
                    "OpenAI response content is None",
                    extra=get_metadata(thread_id=str(self.thread))
                )
                raise ValueError("OpenAI response content is None")

            if isinstance(response.content, list):
                answer_text = "\n".join(str(item) for item in response.content)
            else:
                answer_text = str(response.content)
            return answer_text.strip()
        except Exception as e:
            logger.error(
                f"Error during LLM processing: {str(e)}",
                extra=get_metadata(thread_id=str(self.thread))
            )
            raise e

    def _get_chain_graph(self, user_query, graph_context):
        """
        Get the chain for processing the graph context and user query.
        Args:
            user_query (str): The user's query to be answered
                using the graph context.
            graph_context (dict): A dictionary containing the graph
                context with nodes and edges.
        Returns:
            tuple: A tuple containing:
                - nodes_str (str): A string representation of the nodes.
                - edges_str (str): A string representation of the edges.
                - chain (ChatPromptTemplate): The chain for processing
                    the graph context and user query.
        Raises:
            ValueError: If the graph context is not provided or
                if the user query is empty.
        """

        nodes_str: str = ", ".join(graph_context["nodes"])
        edges_str: str = "; ".join(graph_context["edges"])

        prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    "Provide the answer for the following question:"
                ),
                (
                    "human",
                    AGENT_PROMPT
                )
            ])

        chain = prompt | self.client_llm
        return {
            "nodes_str": nodes_str,
            "edges_str": edges_str,
            "user_query": user_query
        }, chain

    async def _ingestion_batch(
        self,
        documents: list[Document],
        **kwargs
    ) -> AsyncGenerator[Any, Any]:
        """
        Ingest a batch of documents into the memory graph.
        Args:
            documents (list[Document]): A list of Document
            objects to be ingested.
            is_delete_relationships (bool, optional): Whether to delete
                all existing relationships before ingestion.
                Defaults to True.
            limit (int, optional): The maximum number of documents to ingest
                in a single batch.
                Defaults to 0. (No limit, ingest all documents)
            **kwargs: Additional keyword arguments for ingestion configuration.
        Returns:
            None
        Raises:
            ValueError: If the collection name is not provided or
            if the documents list is empty.
        """
        if not documents:
            raise ValueError("No documents provided for ingestion")
        limit = kwargs.get("limit", 0)

        if limit > 0:
            documents = documents[:limit]
            logger.debug(
                f"Limiting ingestion to the first {limit} documents.",
                extra=get_metadata(thread_id=str(self.thread))
            )

        index: int = 1
        for document in documents:
            title = document.metadata.get("title", "Untitled")
            msg = f"Ingesting document {index}/{len(documents)}: {title}"
            logger.debug(f"{msg}: {title}",
                         extra=get_metadata(thread_id=str(self.thread)))
            raw_data = document.page_content
            try:
                async for step in self._ingestion(
                    raw_data=raw_data,
                    metadata=document.metadata
                ):
                    yield f"{step} ({index}/{len(documents)} - {title})"
            except Exception as e:
                logger.error(
                    f"Error during ingestion of document {index}: {str(e)}",
                    extra=get_metadata(thread_id=str(self.thread))
                )
                index += 1
                yield "ERROR"
                continue

            index += 1

    async def _ingestion(self, raw_data, metadata: dict | None = None):
        """
        Ingest data into the memory graph.
        This method should be implemented to handle the ingestion process.
        Args:
            raw_data (str): The raw data to be ingested.
            collection_name (str, optional): The name of the Qdrant
                collection to use for ingestion.
            **kwargs: Additional keyword arguments for ingestion configuration.
        Raises:
            NotImplementedError: If the method is not implemented.
        """
        try:
            yield "Analyzing raw data for graph components."
            nodes, relationships = await self.extract_graph_components(
                raw_data
            )
            yield "Extracted graph components from raw data."

            logger.debug(
                f"Extracted {len(nodes)} nodes and "
                f"{len(relationships)} relationships from the raw data."
            )
            yield "Saving nodes and relationships"
            node_id_mapping = self.ingest_to_neo4j(nodes, relationships)
            logger.debug(
                f"Ingested {len(node_id_mapping)} nodes into Neo4j."
            )
            yield "Vectorizing raw data and ingesting data"

            await self.ingest_to_qdrant(
                raw_data=raw_data,
                node_id_mapping=node_id_mapping,
                metadata=metadata
            )
            yield "Vectorized raw data and ingested data"
            logger.debug(
                f"Ingested data into Qdrant collection "
                f"'{self.collection_name}'."
            )
        except Exception as e:
            logger.error(
                f"Error during ingestion process: {str(e)}",
                extra=get_metadata(thread_id=str(self.thread))
            )
            raise e

    async def query_stream(
        self,
        query: str,
        entity_ids: Optional[list[Any]] = None
    ):
        """
        Query the memory graph using the provided query.
        Args:
            query (str): The query to be executed on the memory graph.
            collection_name (str, optional): The name of the Qdrant
                collection to use for the query.
            entity_ids (Optional[list[Any]]): A list of entity IDs
                to fetch related nodes and relationships.
        Returns:
            list: A list of search results from the memory graph.
        Raises:
            ValueError: If the collection name is not provided
                or if the query is empty.
        """
        try:
            if not query:
                raise ValueError("Query must not be empty")

            graph_context = self._get_graph_context(
                query,
                entity_ids=entity_ids
            )
            async for s in self._stream(
                graph_context=graph_context,
                user_query=query
            ):
                logger.debug(f"Generated answer from LLM: {s}")
                yield s
        except Exception as e:
            logger.error(f"Error during query process: {str(e)}",
                         extra=get_metadata(thread_id=str(self.thread)))
            raise e

    async def query(
        self,
        query: str,
        entity_ids: Optional[list[Any]] = None
    ):
        """
        Query the memory graph using the provided query.
        Args:
            query (str): The query to be executed on the memory graph.
            collection_name (str, optional): The name of the Qdrant
                collection to use for the query.
            entity_ids (Optional[list[Any]]): A list of entity IDs
                to fetch related nodes and relationships.
        Returns:
            list: A list of search results from the memory graph.
        Raises:
            ValueError: If the collection name is not provided
                or if the query is empty.
        """
        try:
            if not query:
                raise ValueError("Query must not be empty")

            graph_context = self._get_graph_context(
                query,
                entity_ids=entity_ids
            )

            # Run the LLM to get the answer
            answer = await self._run(
                graph_context=graph_context,
                user_query=query
            )
            logger.debug(f"Generated answer from LLM: {answer}")
            return answer
        except Exception as e:
            logger.error(f"Error during query process: {str(e)}",
                         extra=get_metadata(thread_id=str(self.thread)))
            raise e

    def retrieve_ids(self, query: str):
        """
        Retrieve entity IDs from the memory graph based on the provided query.
        Args:
            query (str): The query to be executed on the memory graph.
        Returns:
            list: A list of entity IDs retrieved from the memory graph.
        Raises:
            ValueError: If the Neo4j driver is not initialized
                or if the query is empty.
        """
        try:
            if not query:
                raise ValueError("Query must not be empty")

            if not self.neo4j_driver:
                msg: str = "Neo4j driver is not initialized."
                logger.error(
                    msg,
                    extra=get_metadata(thread_id=str(self.thread))
                )
                raise ValueError(msg)

            retriever_result = self.retriever_search(
                query=query,
                neo4j_driver=self.neo4j_driver
            )
            return [
                item.content.split("'id': '")[1].split("'")[0]
                for item in retriever_result.items
            ]
        except Exception as e:
            logger.error(f"Error during retrieval process: {str(e)}",
                         extra=get_metadata(thread_id=str(self.thread)))
            raise e

    def _get_graph_context(
        self,
        query,
        entity_ids: Optional[list[Any]] = None
    ):
        """
        Get the graph context for the provided query.
        Args:
            query (str): The query to be executed on the memory graph.
            entity_ids (Optional[list[Any]]): A list of entity IDs
                to fetch related nodes and relationships.
        Returns:
            dict: A dictionary containing the graph context with
            nodes and edges.
        Raises:
            ValueError: If the Neo4j driver is not initialized
                or if the query is empty.
        """
        if entity_ids is None:
            entity_ids = self.retrieve_ids(query=query)

        logger.debug(
                f"Extracted {len(entity_ids)} entity IDs from the "
                "retriever results."
            )
        subgraph = self._fetch_related_graph(entity_ids=entity_ids)
        logger.debug(
                f"Fetched subgraph with {len(subgraph)} related nodes "
                "and relationships from Neo4j."
            )
        graph_context = self._format_graph_context(subgraph)
        logger.debug(
                f"Formatted graph context with {len(graph_context['nodes'])} "
                f"nodes and {len(graph_context['edges'])} edges."
            )

        return graph_context
