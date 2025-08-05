import uuid
from memory_agent import MemoryPersistence
from qdrant_client.models import PointStruct
from langchain_ollama import OllamaEmbeddings
from neo4j_graphrag.retrievers import QdrantNeo4jRetriever
from langchain_openai import OpenAIEmbeddings
from log import logger, get_metadata


class KGragVectorStore(MemoryPersistence):
    """
    MemoryVector is a class that manages a memory vector store using
    Qdrant as the vector store.
    It provides methods to add messages, delete collections,
    and retrieve data.
    """

    def __init__(self, **kwargs):
        """
        Initialize the MemoryStoreVector with the provided parameters.
        Args:
            **kwargs: Additional keyword arguments for configuration.
        """
        super().__init__(**kwargs)

    async def ingest_to_qdrant(
        self,
        raw_data,
        node_id_mapping,
        metadata: dict | None = None
    ):
        """
        Ingest raw data into Qdrant as embeddings.
        Args:
            raw_data (str): The raw data to be ingested.
            node_id_mapping (dict): A mapping of node names to unique IDs.
            collection_name (str, optional): The name of the Qdrant collection.
                If not provided, the default collection name will be used.
            metadata (dict, optional): Additional metadata to be added
                to the payload of each point.
                Defaults to None.
        Returns:
            None
        Raises:
            ValueError: If the collection name is not provided
                or if the raw data is empty.
        """
        try:
            openai_embeddings = self.get_embedding_model()

            if not self.collection_name:
                raise ValueError("Collection name must be provided")

            if await self.create_collection_async(
                self.collection_name,
                self.collection_dim
            ):
                logger.debug(
                    f"Collection '{self.collection_name}' created successfully"
                )

            if openai_embeddings is None:
                msg: str = (
                    "Embedding model is not set or failed to load"
                )
                logger.error(
                    msg,
                    extra=get_metadata(thread_id=str(self.thread))
                )
                raise ValueError(msg)

            if not (
                isinstance(openai_embeddings, OpenAIEmbeddings) or
                (
                    OllamaEmbeddings and
                    isinstance(openai_embeddings, OllamaEmbeddings)
                )
            ):
                raise ValueError(
                    "Embedding model must be an instance of "
                    "OpenAIEmbeddings or OllamaEmbeddings"
                )

            embeddings = [
                openai_embeddings.embed_query(paragraph)
                for paragraph in raw_data.split("\n")
            ]

            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={"id": node_id}
                )
                for node_id, embedding in zip(
                    node_id_mapping.values(),
                    embeddings
                )
            ]

            # Add metadata to each point's payload if provided
            if metadata:
                for point in points:
                    if isinstance(point.payload, dict):
                        point.payload.update(metadata)
                    else:
                        point.payload = metadata

            await self.qdrant_client_async.upsert(
                collection_name=self.collection_name,
                points=points
            )
        except Exception as e:
            logger.error(
                f"Error during Qdrant ingestion: {str(e)}",
                extra=get_metadata(thread_id=str(self.thread))
            )
            raise e

    def retriever_search(self, query, neo4j_driver):
        """
        Perform a search using the QdrantNeo4jRetriever.
        Args:
            neo4j_driver (Driver): The Neo4j driver instance.
            qdrant_client (QdrantClient): The Qdrant client instance.
            collection_name (str): The name of the Qdrant collection.
            query (str): The search query.
        Returns:
            list: A list of search results.
        Raises:
            ValueError: If the Neo4j driver or Qdrant client
                is not initialized.
        """
        try:
            retriever = QdrantNeo4jRetriever(
                driver=neo4j_driver,
                client=self.qdrant_client,
                collection_name=self.collection_name,
                id_property_external="id",
                id_property_neo4j="id",
            )

            openai_embeddings = self.get_embedding_model()

            if not (
                isinstance(openai_embeddings, OpenAIEmbeddings) or
                (
                    OllamaEmbeddings and
                    isinstance(openai_embeddings, OllamaEmbeddings)
                )
            ):
                msg: str = (
                    "Embedding model must be an instance of "
                    "OpenAIEmbeddings or OllamaEmbeddings"
                )
                logger.error(
                    msg,
                    extra=get_metadata(thread_id=str(self.thread))
                )
                raise ValueError(msg)

            query_vector = openai_embeddings.embed_query(query)
            results = retriever.search(
                query_vector=query_vector,
                top_k=5
            )

            return results
        except Exception as e:
            logger.error(
                f"Error during retriever search: {str(e)}",
                extra=get_metadata(thread_id=str(self.thread))
            )
            raise e
