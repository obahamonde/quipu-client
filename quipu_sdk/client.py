from __future__ import annotations
import os
import httpx
from typing import Generic, TypeVar, Type, Any
from pydantic import Field
from .schemas import Base, RagRequest, CosimResult, Status, QuipuModel
from .proxy import Proxy
from .utils import handle


T = TypeVar("T", bound=Base)


class QuipuClient(Base, Proxy[httpx.AsyncClient], Generic[T]):

    base_url: str = Field(
        default=os.getenv("QUIPUBASE_URL", "https://db.indiecloud.co/")
    )

    headers: dict[str, str] = Field(
        default_factory=lambda: {"Content-Type": "application/json"}
    )

    @classmethod
    def __class_getitem__(cls, item: Type[T]):  # type: ignore
        cls._model = item
        return cls

    def __load__(self):
        return httpx.AsyncClient(base_url=self.base_url, headers=self.headers)

    @handle
    async def put(self, *, namespace: str, instance: T):
        response = await self.__load__().post(
            f"/api/document/{namespace}?action=put",
            json=QuipuModel[self._model](data=instance).model_dump(),
        )
        response.raise_for_status()
        data = response.json()
        assert self._model
        return self._model(**data)

    @handle
    async def get(self, *, namespace: str, key: str):
        response = await self.__load__().post(
            f"/api/document/{namespace}?action=get&key={key}",
            json=QuipuModel[self._model]().model_dump(),
        )
        response.raise_for_status()
        data = response.json()
        try:
            assert self._model
            return self._model(**data)
        except:
            return Status(**data)

    @handle
    async def merge(self, *, namespace: str, instance: T):
        response = await self.__load__().post(
            f"/api/document/{namespace}?action=merge",
            json=QuipuModel[self._model](data=instance).model_dump(),
        )
        response.raise_for_status()
        _json = response.json()
        assert self._model
        return self._model(**_json)

    @handle
    async def delete(self, *, namespace: str, key: str):
        response = await self.__load__().post(
            f"/api/document/{namespace}?action=delete&key={key}",
            json=QuipuModel[self._model]().model_dump(),
        )
        response.raise_for_status()
        return Status(**response.json())

    @handle
    async def find(self, *, namespace: str, **kwargs: Any):
        response = await self.__load__().post(
            f"/api/document/{namespace}?action=find",
            json={
                "data": kwargs,
                "definition": {
                    "type": "object",
                    "title": self._model.__name__,
                    "description": self._model.__doc__ or "[No description]",
                    "properties": self._model.model_json_schema().get("properties", {}),
                },
            },
        )
        response.raise_for_status()
        assert self._model
        return [self._model(**d) for d in response.json()]

    @handle
    async def upsert(self, namespace: str, data: RagRequest):
        response = await self.__load__().post(
            f"/api/vector/{namespace}?action=upsert", json=data
        )
        response.raise_for_status()
        return Status(code=201, message="Upserted")

    @handle
    async def query(self, namespace: str, data: RagRequest, top_k: int):
        response = await self.__load__().post(
            f"/api/vector/{namespace}?action=query&topK={top_k}", json=data
        )
        response.raise_for_status()
        res = response.json()
        return [CosimResult(**d) for d in res]

    @handle
    async def health_check(self):
        response = await self.__load__().get("/api/health")
        response.raise_for_status()
        return Status(code=200, message="Healthy")


QuipuClient.model_rebuild()
