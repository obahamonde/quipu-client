import pytest
import httpx
from quipu_sdk import QuipuClient, Base, RagRequest


class TestModel(Base):
    name: str
    age: int


@pytest.fixture
def mock_quipu_client():
    return QuipuClient[TestModel](_model=TestModel)


@pytest.mark.asyncio
async def test_put_document(mock_quipu_client, mocker):
    async def mock_post(*args, **kwargs):
        return httpx.Response(200, json={"key": "123", "name": "John", "age": 30})

    mocker.patch("httpx.AsyncClient.post", side_effect=mock_post)

    instance = TestModel(name="John", age=30)
    result = await mock_quipu_client.put(namespace="test_namespace", instance=instance)

    assert result.key == "123"
    assert result.name == "John"
    assert result.age == 30


@pytest.mark.asyncio
async def test_get_document(mock_quipu_client, mocker):
    async def mock_post(*args, **kwargs):
        return httpx.Response(200, json={"key": "123", "name": "John", "age": 30})

    mocker.patch("httpx.AsyncClient.post", side_effect=mock_post)

    result = await mock_quipu_client.get(namespace="test_namespace", key="123")

    assert result.key == "123"
    assert result.name == "John"
    assert result.age == 30


@pytest.mark.asyncio
async def test_merge_document(mock_quipu_client, mocker):
    async def mock_post(*args, **kwargs):
        return httpx.Response(200, json={"key": "123", "name": "John", "age": 31})

    mocker.patch("httpx.AsyncClient.post", side_effect=mock_post)

    instance = TestModel(name="John", age=31)
    result = await mock_quipu_client.merge(
        namespace="test_namespace", instance=instance
    )

    assert result.key == "123"
    assert result.name == "John"
    assert result.age == 31


@pytest.mark.asyncio
async def test_delete_document(mock_quipu_client, mocker):
    async def mock_post(*args, **kwargs):
        return httpx.Response(200, json={"code": 200, "message": "Deleted"})

    mocker.patch("httpx.AsyncClient.post", side_effect=mock_post)

    result = await mock_quipu_client.delete(namespace="test_namespace", key="123")

    assert result.code == 200
    assert result.message == "Deleted"


@pytest.mark.asyncio
async def test_find_documents(mock_quipu_client, mocker):
    async def mock_post(*args, **kwargs):
        return httpx.Response(200, json=[{"key": "123", "name": "John", "age": 30}])

    mocker.patch("httpx.AsyncClient.post", side_effect=mock_post)

    result = await mock_quipu_client.find(namespace="test_namespace", name="John")

    assert len(result) == 1
    assert result[0].key == "123"
    assert result[0].name == "John"
    assert result[0].age == 30


@pytest.mark.asyncio
async def test_query_vector(mock_quipu_client, mocker):
    async def mock_post(*args, **kwargs):
        return httpx.Response(
            200, json=[{"id": "1", "score": 0.9, "content": "example content"}]
        )

    mocker.patch("httpx.AsyncClient.post", side_effect=mock_post)

    data = RagRequest(content="example query")
    result = await mock_quipu_client.query(
        namespace="test_namespace", data=data, top_k=5
    )

    assert len(result) == 1
    assert result[0].id == "1"
    assert result[0].score == 0.9
    assert result[0].content == "example content"


@pytest.mark.asyncio
async def test_upsert_vector(mock_quipu_client, mocker):
    async def mock_post(*args, **kwargs):
        return httpx.Response(201, json={"code": 201, "message": "Upserted"})

    mocker.patch("httpx.AsyncClient.post", side_effect=mock_post)

    data = RagRequest(content="example query")
    result = await mock_quipu_client.upsert(namespace="test_namespace", data=data)

    assert result.code == 201
    assert result.message == "Upserted"


@pytest.mark.asyncio
async def test_health_check(mock_quipu_client, mocker):
    async def mock_get(*args, **kwargs):
        return httpx.Response(200, json={"code": 200, "message": "Healthy"})

    mocker.patch("httpx.AsyncClient.get", side_effect=mock_get)

    result = await mock_quipu_client.health_check()

    assert result.code == 200
    assert result.message == "Healthy"
