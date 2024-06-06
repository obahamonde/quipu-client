# quipu_sdk/proxy.py

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Iterable, TypeVar, cast
from pydantic import BaseModel, create_model  # type: ignore
from typing import Type, Optional, TypeVar, Generic

from typing_extensions import override

T = TypeVar("T")
T_ = TypeVar("T_", bound=BaseModel)


class Proxy(Generic[T], ABC):
    """Implements data methods to pretend that an instance is another instance.

    This includes forwarding attribute access and other methods.
    """

    # Note: we have to special case proxies that themselves return proxies
    # to support using a proxy as a catch-all for any random access, e.g. `proxy.foo.bar.baz`

    def __getattr__(self, attr: str) -> object:
        proxied = self.__get_proxied__()
        if isinstance(proxied, Proxy):
            return proxied  # pyright: ignore
        return getattr(proxied, attr)

    @override
    def __repr__(self) -> str:
        proxied = self.__get_proxied__()
        if isinstance(proxied, Proxy):
            return proxied.__class__.__name__
        return repr(self.__get_proxied__())

    @override
    def __str__(self) -> str:
        proxied = self.__get_proxied__()
        if isinstance(proxied, Proxy):
            return proxied.__class__.__name__
        return str(proxied)

    @override
    def __dir__(self) -> Iterable[str]:
        proxied = self.__get_proxied__()
        if isinstance(proxied, Proxy):
            return []
        return proxied.__dir__()

    @property  # type: ignore
    @override
    def __class__(self) -> type:  # pyright: ignore
        proxied = self.__get_proxied__()
        if issubclass(type(proxied), Proxy):
            return type(proxied)
        return proxied.__class__

    def __get_proxied__(self) -> T:
        return self.__load__()

    def __as_proxied__(self) -> T:
        """Helper method that returns the current proxy, typed as the loaded object"""
        return cast(T, self)

    @abstractmethod
    def __load__(self) -> T: ...


def create_partial_model(base_model: Type[T]) -> Type[T]:
    """
    Create a new model with all fields optional based on the provided base model.
    """
    fields = {
        name: (Optional[typ], None) for name, typ in base_model.__annotations__.items()
    }
    partial_cls = create_model(f"Partial{base_model.__name__}", **fields)  # type: ignore
    return partial_cls  # type: ignore


class Partial(Generic[T_]):
    """
    Partial class for models, creating a version where all fields are optional.
    """

    @classmethod
    def __class_getitem__(cls, item: Type[T_]) -> Type[T_]:
        return create_partial_model(item)

```
---
# quipu_sdk/schemas.py

```python
from functools import cached_property
from pydantic import BaseModel, Field, computed_field
from typing import Union, Literal, TypeVar, Generic, Type, Optional
from typing_extensions import TypedDict
from uuid import uuid4


class Base(BaseModel):
    key: str = Field(default_factory=lambda: str(uuid4()))

    def __str__(self) -> str:
        return self.model_dump_json()

    def __repr__(self) -> str:
        return self.model_dump_json()


T = TypeVar("T", bound=Base)


class JsonSchema(TypedDict, total=False):
    title: str
    description: str
    type: Literal["object"]
    properties: dict[str, object]


class QuipuModel(BaseModel, Generic[T]):

    data: Optional[T] = Field(default=None)

    @computed_field
    @cached_property
    def definition(self) -> JsonSchema:
        return JsonSchema(
            type="object",
            title=self._model.__name__,
            description=self._model.__doc__ or "[No description]",
            properties=self._model.model_json_schema().get("properties", {}),
        )

    @classmethod
    def __class_getitem__(cls, item: Type[T]):  # type: ignore
        cls._model = item
        return cls


class Status(BaseModel):
    code: int
    message: str
    key: Optional[str] = Field(default=None)
    definition: Optional[JsonSchema] = Field(default=None)


class CosimResult(BaseModel):
    id: str
    score: float
    content: Union[str, list[str], list[float]]


class RagRequest(TypedDict, total=False):
    content: Union[str, list[str], list[float]]

```
---
# quipu_sdk/__init__.py

```python
from .client import QuipuClient, Base, RagRequest, CosimResult, Status

__all__ = ["QuipuClient", "Base", "RagRequest", "CosimResult", "Status"]

```
---
# quipu_sdk/utils.py

```python
from __future__ import annotations
import asyncio
import json
import logging
import time
from functools import partial, wraps
from typing import Awaitable, Callable, Coroutine, Type, TypeVar, Union, cast

from fastapi import HTTPException
from typing_extensions import ParamSpec

T = TypeVar("T")
P = ParamSpec("P")


def get_logger(
    name: str | None = None,
    level: int = logging.DEBUG,
    format_string: str = json.dumps(
        {
            "timestamp": "%(asctime)s",
            "level": "%(levelname)s",
            "name": "%(name)s",
            "message": "%(message)s",
        }
    ),
) -> logging.Logger:
    """
    Configures and returns a logger with a specified name, level, and format.

    :param name: Name of the logger. If None, the root logger will be configured.
    :param level: Logging level, e.g., logging.INFO, logging.DEBUG.
    :param format_string: Format string for log messages.
    :return: Configured logger.
    """
    if name is None:
        name = "QuipuBase 🚀"
    logger_ = logging.getLogger(name)
    logger_.setLevel(level)
    if not logger_.handlers:
        ch = logging.StreamHandler()
        formatter = logging.Formatter(format_string)
        ch.setFormatter(formatter)
        logger_.addHandler(ch)
    return logging.getLogger(name)


logger = get_logger()


def exception_handler(
    func: Callable[P, T]
) -> Callable[P, Union[T, Coroutine[None, T, T]]]:
    """
    Decorator to handle exceptions in a function.

    :param func: Function to be decorated.
    :return: Decorated function.
    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error("%s: %s", e.__class__.__name__, e)
            raise HTTPException(
                status_code=500,
                detail=f"Internal Server Error: {e.__class__.__name__} => {e}",
            ) from e

    async def awrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            func_ = cast(Awaitable[T], func(*args, **kwargs))
            return await func_
        except Exception as e:
            logger.error("%s: %s", e.__class__.__name__, e)
            raise HTTPException(
                status_code=500,
                detail=f"Internal Server Error: {e.__class__.__name__} => {e}",
            ) from e

    if asyncio.iscoroutinefunction(func):
        awrapper.__name__ = func.__name__
        return awrapper
    wrapper.__name__ = func.__name__
    return wrapper


def timing_handler(
    func: Callable[P, T]
) -> Callable[P, Union[T, Coroutine[None, T, T]]]:
    """
    Decorator to measure the time taken by a function.

    :param func: Function to be decorated.
    :return: Decorated function.
    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info("%s took %s seconds", func.__name__, end - start)
        return result

    async def awrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        start = time.time()
        func_ = cast(Awaitable[T], func(*args, **kwargs))
        result = await func_
        end = time.time()
        logger.info("%s took %s seconds", func.__name__, end - start)
        return result

    if asyncio.iscoroutinefunction(func):
        awrapper.__name__ = func.__name__
        return awrapper
    wrapper.__name__ = func.__name__
    return wrapper


def retry_handler(
    func: Callable[P, T], retries: int = 3, delay: int = 1
) -> Callable[P, Union[T, Coroutine[None, T, T]]]:
    """
    Decorator to retry a function with exponential backoff.

    :param func: Function to be decorated.
    :param retries: Number of retries.
    :param delay: Delay between retries.
    :return: Decorated function.
    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        nonlocal delay
        try:
            for _ in range(retries):
                try:
                    return func(*args, **kwargs)
                except HTTPException as e:
                    setattr(func, "exception", e)
                    logger.error("%s: %s", e.__class__.__name__, e)
                    time.sleep(delay)
                    delay *= 2
            raise HTTPException(
                status_code=500,
                detail=f"Exhausted retries: {func.exception.__class__.__name__} => {func.exception}",  # type: ignore
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Internal Server Error: {e.__class__.__name__} => {e}",
            ) from e

    @wraps(func)
    async def awrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        nonlocal delay
        for _ in range(retries):
            try:
                func_ = cast(Awaitable[T], func(*args, **kwargs))
                return await func_
            except (
                Exception,
                AssertionError,
                KeyError,
                ValueError,
                HTTPException,
            ) as e:
                logger.error("%s: %s", e.__class__.__name__, e)
                await asyncio.sleep(delay)
                delay *= 2
        raise HTTPException(
            status_code=500,
            detail="Exhausted retries",
        )

    if asyncio.iscoroutinefunction(func):
        awrapper.__name__ = func.__name__
        return awrapper
    wrapper.__name__ = func.__name__
    return wrapper


def handle(
    func: Callable[P, T], retries: int = 3, delay: int = 1
) -> Callable[P, Union[T, Coroutine[None, T, T]]]:
    """
    Decorator to retry a function with exponential backoff and handle exceptions.

    :param func: Function to be decorated.
    :param retries: Number of retries.
    :param delay: Delay between retries.
    :return: Decorated function.
    """
    eb = partial(retry_handler, retries=retries, delay=delay)
    return cast(
        Callable[P, Union[T, Coroutine[None, T, T]]],
        timing_handler(exception_handler(eb(func))),
    )


def asyncify(func: Callable[P, T]) -> Callable[P, Coroutine[None, T, T]]:
    """
    Decorator to convert a synchronous function to an asynchronous function.

    :param func: Synchronous function to be decorated.
    :return: Asynchronous function.
    """

    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        wrapper.__name__ = func.__name__
        return await asyncio.to_thread(func, *args, **kwargs)

    return wrapper


def singleton(cls: Type[T]) -> Type[T]:
    """
    Decorator that converts a class into a singleton.

    Args:
                                                                                                                                                                                                                                                                    cls (Type[T]): The class to be converted into a singleton.

    Returns:
                                                                                                                                                                                                                                                                    Type[T]: The singleton instance of the class.
    """
    instances: dict[Type[T], T] = {}

    @wraps(cls)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return cast(Type[T], wrapper)

```
---
# quipu_sdk/client.py

```python
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

```
---
# tests/test_client.py

```python
import pytest
from quipu_sdk import QuipuClient, Base


class MockModel(Base):
    name: str
    age: int


@pytest.fixture(scope="module")
def mock_quipu_client():
    return QuipuClient[MockModel]()


@pytest.mark.parametrize(
    "name, age",
    (
        ("Alice", 20),
        ("Bob", 30),
        ("Charlie", 40),
        ("David", 50),
        ("Eve", 60),
    ),
)
@pytest.mark.asyncio
async def test_crud_operations(
    mock_quipu_client: QuipuClient[MockModel], name: str, age: int
):
    instance = MockModel(name=name, age=age)
    response = await mock_quipu_client.put(namespace="test", instance=instance)
    assert response.name == name
    assert response.age == age
    assert instance.key == response.key
    response = await mock_quipu_client.get(namespace="test", key=response.key)
    assert response.name == name  # type: ignore
    assert response.age == age  # type: ignore
    instance = MockModel(name=name, age=age + 10)
    await mock_quipu_client.put(namespace="test", instance=instance)
    response = await mock_quipu_client.find(namespace="test")
    assert isinstance(response[0], MockModel)
    response = await mock_quipu_client.merge(namespace="test", instance=instance)
    assert response.name == name
    assert response.age == age + 10
    response = await mock_quipu_client.delete(namespace="test", key=response.key)
    assert response.code == 204
    response = await mock_quipu_client.get(namespace="test", key=response.key)  # type: ignore
    assert response.code == 404  # type: ignore


@pytest.mark.parametrize(
    "content",
    (
        "The lazy dog jumped over the quick brown fox",
        "The quick brown fox jumped over the lazy dog",
        "The yellow cat jumped over the lazy lion",
        "The joker is a good movie",
        "The man in the middle is a heuristic lorem ipsum",
        "Pizza is the best food",
        "Python is the best programming language",
        "Javascript is the worst programming language",
        "AGI is the future",
        "ChatGPT is the best AI model",
        "OpenAI is the best AI company",
        "Google is the Devil",
        "This is sparta",
        "Hello World",
    ),
)
@pytest.mark.asyncio
async def test_upsert_vector(mock_quipu_client: QuipuClient[MockModel], content: str):
    response = await mock_quipu_client.upsert(
        namespace="test", data={"content": content}
    )
    assert response.code == 201


@pytest.mark.parametrize(
    "content",
    (
        "I and my friends went to the park",
        "This is Sparta",
        "Java is the best programming language",
        "Python is the worst programming language",
        "AGI is the future",
        "Do you hear the music?",
    ),
)
@pytest.mark.asyncio
async def test_query_vector(mock_quipu_client: QuipuClient[MockModel], content: str):
    response = await mock_quipu_client.query(
        namespace="test", data={"content": content}, top_k=5
    )
    assert len(response) == 5
    assert response[0].score - 1.0 < 0.01
    assert response[0].id

```
---
# tests/__init__.py

```python

```
---
