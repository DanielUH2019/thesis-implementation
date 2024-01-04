from typing import Optional


from pydantic import BaseModel
from typing import Union, Any
from sentence_transformers import SentenceTransformer
from uuid import uuid4
import chromadb


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
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(collection_name)
        # Given a key retrieved from the vector database, this dictionary will return his value
        # This value can be a number, a function, a list, etc
        self.value_store: dict[str, Any] = {}
        # Given a key, this dictionary will return the id of the vector in the vector database
        self._id_store: dict[str, Union[int, str]] = {}
        self._all_ids = []

    def get_all_data(self):
        return self.collection.get()

    def upload_records(self, ids, documents, metadatas):
        self.collection.add(ids=ids, documents=documents, metadatas=metadatas)

    def query(
        self, query: str, query_filter: Optional[chromadb.Where] = None, limit: int = 1
    ):
        results = self.collection.query(
            query_texts=[query], where=query_filter, n_results=limit
        )
        return results

    def retrieve_stored_value(self, description: str):
        # if key in self.value_store:
        #     return self.value_store[key]

        search_result = self.query(query=description, limit=1)
        if len(search_result["ids"][0]) == 0:
            # print("all: ", self.get_all_data())
            raise ValueError(
                f"Could not find a similar values for description {description}"
            )

        most_similar_key = search_result["metadatas"][0][0]["key_name"]
        most_similar_id = search_result["ids"][0][0]
        if most_similar_id not in self.value_store:
            # print(f" value store {self.value_store}")
            raise ValueError(
                f"The most similar key {most_similar_key} with description {search_result['documents'][0][0]} is not in the value store"
            )

        return most_similar_id, most_similar_key, self.value_store[most_similar_id]

    def insert_to_value_store(self, data: list[ValueStoreObjectModel]):
        ids = [str(uuid4()) for i in data]
        self.collection.add(
            documents=[x.description for x in data],
            metadatas=[{"key_name": x.key_name} for i, x in enumerate(data)],
            ids=ids,
        )
        # print(f"mmmm", print(self.get_all_data()))
        for i in range(len(data)):
            self.value_store[ids[i]] = data[i].value
            self._id_store[data[i].key_name] = ids[i]

    def insert(self, documents, metadatas) -> list[str]:
        if len(documents) == 0:
            return []

        ids = [str(uuid4()) for d in documents]
        self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
        return ids

    def update(self, id: str, data: str):
        self.collection.upsert(ids=[id], documents=[data])

    def delete_from_store(self, ids_to_delete: list[str]):
        self.collection.delete(ids=ids_to_delete)
        for id in ids_to_delete:
            del self.value_store[id]

    def delete_from_db(self, id: str):
        self.collection.delete(ids=[id])
