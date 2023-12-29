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
    
    def query(self, query: str, query_filter: Optional[chromadb.Where] = None, limit: int = 1):
        # print("mmmm", self.get_all_data())
        # print('query: ', query)
        results = self.collection.query(
            query_texts=[query],
            where=query_filter,
            n_results=limit
        )
        # print(f"query results {results}")
        return results

    def retrieve_stored_value(self, key: str, description: str):
        if key in self.value_store:
            return self.value_store[key]

        search_result = self.query(query=description, limit=1)
        if len(search_result['ids'][0]) == 0:
            raise ValueError(
                f"Could not find a similar values for key {key} and description: {description}"
            )

        most_similar_key = search_result['metadatas'][0][0]["key_name"]
        if most_similar_key not in self.value_store:
            raise ValueError(
                f"The most similar key {most_similar_key} is not in the value store"
            )

        return self.value_store[most_similar_key]

    def insert_to_value_store(self, data: list[ValueStoreObjectModel]):
        ids = [str(uuid4()) for i in data]
        self.collection.add(
            documents=[x.description for x in data],
            metadatas=[{"key_name": x.key_name} for x in data],
            ids=ids
        )
        for i in range(len(data)):
            self.value_store[data[i].key_name] = data[i].value
            self._id_store[data[i].key_name] = ids[i]

    def insert(self, documents, metadatas) -> list[str]:
        if len(documents) == 0:
            return []
        
        # print(f'documents to insert {documents}')
        ids = [str(uuid4()) for d in documents]
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        # print(f'all_data {self.get_all_data()}')
        return ids
    
    def update(self, id: str, data: str):
        self.collection.upsert(ids=[id], documents=[data])


    def delete_from_store(self, key: str):
        id_to_delete = self._id_store[key]
        self.collection.delete(
            ids=[id_to_delete]
        )
        del self.value_store[key]

    def delete_from_db(self, id: str):
        self.collection.delete(
            ids=[id]
        )
