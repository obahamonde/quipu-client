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


class QuipuModel(Base, Generic[T]):

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


class Status(Base):
    code: int
    message: str
    key: Optional[str] = Field(default=None)
    definition: Optional[JsonSchema] = Field(default=None)


class CosimResult(Base):
    id: str
    score: float
    content: Union[str, list[str], list[float]]


class RagRequest(TypedDict, total=False):
    content: Union[str, list[str], list[float]]
