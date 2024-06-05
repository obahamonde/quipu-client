from __future__ import annotations
import os
import httpx
from typing import TypeVar, Type, Any, Generic
from dataclasses import dataclass, field
from .schemas import Base, RagRequest, CosimResult, Status
from .proxy import Proxy


T = TypeVar("T", bound=Base)


@dataclass
class QuipuClient(Proxy[httpx.AsyncClient], Generic[T]):

    def __init__(self, _model: Type[T] | None = None):
        self._model = _model

    def __post_init__(self):
        if self._model is None:
            raise ValueError("Model is required")

    base_url: str = field(
        default=os.getenv("QUIPUBASE_URL", "https://db.indiecloud.co/")
    )

    headers: dict[str, str] = field(
        default_factory=lambda: {"Content-Type": "application/json"}
    )

    def __load__(self):
        return httpx.AsyncClient(base_url=self.base_url, headers=self.headers)

    async def put(self, *, namespace: str, instance: T, action: str):
        response = await self.__load__().post(
            f"/api/document/{namespace}?action=put",
            json=instance.model_dump(),
        )
        response.raise_for_status()
        data = response.json()
        assert self._model
        return self._model(**data)

    async def get(self, *, namespace: str, key: str, action: str):
        response = await self.__load__().post(
            f"/api/document/{namespace}?action=get&key={key}"
        )
        response.raise_for_status()
        data = response.json()
        if data.get("data"):
            assert self._model
            return self._model(**data)
        return Status(code=404, message="Not Found")

    async def merge(self, *, namespace: str, instance: T, action: str):
        response = await self.__load__().post(
            f"/api/document/{namespace}?action=merge", json=instance.model_dump()
        )
        response.raise_for_status()
        _json = response.json()
        assert self._model
        return self._model(**_json)

    async def delete(self, *, namespace: str, key: str, action: str):
        response = await self.__load__().post(
            f"/api/document/{namespace}?action=delete&key={key}"
        )
        response.raise_for_status()
        return Status(code=200, message="Deleted")

    async def find(self, *, namespace: str, action: str, **kwargs: Any):
        response = await self.__load__().post(
            f"/api/document/{namespace}?action=find", json=kwargs
        )
        response.raise_for_status()
        assert self._model
        return [self._model(**d) for d in response.json()]

    async def query(self, namespace: str, data: RagRequest, top_k: int, action: str):
        response = await self.__load__().post(
            f"/api/vector/{namespace}?action=search&topK={top_k}",
            json=data.model_dump(),
        )
        response.raise_for_status()
        res = response.json()
        return [CosimResult(**d) for d in res]

    async def upsert(self, namespace: str, data: RagRequest, action: str):
        response = await self.__load__().post(
            f"/api/vector/{namespace}?action=upsert", json=data.model_dump()
        )
        response.raise_for_status()
        return Status(code=201, message="Upserted")

    async def health_check(self, action: str):
        response = await self.__load__().get("/api/health")
        response.raise_for_status()
        return Status(code=200, message="Healthy")
