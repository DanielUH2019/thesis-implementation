from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, PointStruct, FieldCondition, PointIdsList
from pydantic import BaseModel
from typing import Union, Any


class MemoryDataModel(BaseModel):
    data: str
    key_name: str
    value: Union[int, float, str, list, dict, tuple, set, bool]


class ValueStoreObjectModel(BaseModel):
    key_name: str
    description: str
    value: Union[int, float, str, list, dict, tuple, set, bool]


class Memory:
    def __init__(self, collection_name: str = "memory") -> None:
        self.collection_name = collection_name
        self.client = self._create_vector_db()
        # Given a key retrieved from the vector database, this dictionary will return his value
        # This value can be a number, a function, a list, etc
        self.value_store: dict[str, Any] = {}
        # Given a key, this dictionary will return the id of the vector in the vector database
        self._id_store: dict[str, Union[int, str]] = {}

    def _create_vector_db(self) -> QdrantClient:
        client = QdrantClient(":memory:")
        client.create_collection(
            collection_name=self.collection_name,
            vectors_config=self.client.get_fastembed_vector_params(),
        )
        return client

    def query(self, query: str, query_filter: Optional[Filter] = None, limit: int = 5):
        return self.client.query(
            collection_name="memory",
            query_text=query,
            query_filter=query_filter,
        )

    def retrieve_stored_value(self, key: str, description: str):
        if key in self.value_store:
            return self.value_store[key]

        search_result = self.query(query=description, limit=1)
        if len(search_result) == 0:
            raise ValueError(
                f"Could not find a similar values for key {key} and description: {description}"
            )

        most_similar_key = search_result[0].metadata["key_name"]
        if most_similar_key not in self.value_store:
            raise ValueError(
                f"The most similar key {most_similar_key} is not in the value store"
            )

        return self.value_store[most_similar_key]

    def insert_to_value_store(self, data: list[ValueStoreObjectModel]):
        ids = self.client.add(
            collection_name="value_store",
            documents=[x.description for x in data],
            metadata=[{"key_name": x.key_name} for x in data],
        )
        for i in range(len(data)):
            self.value_store[data[i].key_name] = data[i].value
            self._id_store[data[i].key_name] = ids[i]

    def insert(self, data: list[str]):
        self.client.add(
            collection_name=self.collection_name,
            documents=[x for x in data],
        )

    def delete_from_store(self, key: str):
        id_to_delete = self._id_store[key]
        self.client.delete(
            self.collection_name, points_selector=PointIdsList(points=[id_to_delete])
        )
        del self.value_store[key]
