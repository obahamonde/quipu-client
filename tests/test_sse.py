import pytest
from quipu_sdk import QuipuClient, Base
import asyncio


class JobPosting(Base):
    company: dict
    location: str
    modality: str
    remote: bool
    salary: int
    skills: list[str]
    title: str


@pytest.fixture(scope="module")
def quipu_client():
    return QuipuClient[JobPosting]()


@pytest.mark.skip(reason="Not finished")
@pytest.mark.asyncio
async def test_subscription(quipu_client: QuipuClient[JobPosting]):
    namespace = "jobposting"

    # Create a new JobPosting document
    job_data = JobPosting(
        company={"name": "Acme Inc.", "url": "https://acme.com"},
        location="Remote",
        modality="full-time",
        remote=True,
        salary=100000,
        skills=["python", "fastapi", "aws"],
        title="JobPosting",
    )
    created_doc = await quipu_client.put(namespace=namespace, instance=job_data)

    # Subscribe to changes
    async def listen_to_updates():
        updates = []
        async for event in quipu_client.subscribe(
            namespace=namespace, subscriber="test-subscriber"
        ):
            updates.append(event)
            if len(updates) >= 1:
                break
        return updates

    # Start listening to updates
    updates_task = asyncio.create_task(listen_to_updates())

    # Modify the document
    updated_data = JobPosting(
        company={"name": "Acme Inc.", "url": "https://acme.com"},
        location="Remote",
        modality="full-time",
        remote=True,
        salary=110000,
        skills=["python", "fastapi", "aws", "docker"],
        title="Senior JobPosting",
    )
    await quipu_client.merge(namespace=namespace, instance=updated_data)

    # Wait for the subscription to receive the update
    updates = await updates_task

    # Verify the received updates
    assert len(updates) == 1
    assert updates[0]["data"]["salary"] == 110000
    assert updates[0]["data"]["skills"] == ["python", "fastapi", "aws", "docker"]
    assert updates[0]["data"]["title"] == "Senior JobPosting"
