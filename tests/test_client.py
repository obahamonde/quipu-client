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
