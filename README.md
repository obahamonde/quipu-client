# QuipuBase Client for Python

Welcome to the QuipuBase Client for Python! This SDK is designed to provide a seamless and efficient way to interact with QuipuBase, an advanced data management system tailored for modern AI applications. With QuipuBase, you can leverage high-performance storage, semantic search, custom data modeling, and seamless integrations. Our SDK ensures that developers have all the tools they need to build powerful, scalable, and efficient applications.

## Features

- **High-Performance Storage**: QuipuBase leverages cutting-edge technologies to provide rapid storage and retrieval capabilities, ensuring that your applications run smoothly and efficiently.
- **Semantic Search**: Perform advanced searches with ease using our robust semantic search functionalities.
- **Custom Data Modeling**: Define and manage custom data models to suit the unique needs of your application.
- **Seamless Integrations**: Integrate with various third-party services and platforms effortlessly.

## Installation

To install the QuipuBase Client for Python, simply use pip:

```bash
pip install quipu-sdk
```

# Getting Started

## Initializing the Client

First, import the necessary modules and initialize the client:

```python

from quipubase_client import QuipuClient, Base
from pydantic import BaseModel

# Define your data model
class MyDataModel(BaseModel):
    name: str
    value: int

# Initialize the client with your data model
client = QuipuClient[MyDataModel]()
```

## Inserting Data

To insert data into QuipuBase, use the `put` method:

```python

data = MyDataModel(name="example", value=42)
response = await client.put(namespace="my_namespace", instance=data)
print(response)

```

## Retrieving Data
To retrieve data from QuipuBase, use the `get` method:

```python

key = "your_data_key"
response = await client.get(namespace="my_namespace", key=key)
print(response)

```

# Merging Data

To merge data into an existing entry, use the `merge` method:

```python

data = MyDataModel(name="example_updated", value=43)
response = await client.merge(namespace="my_namespace", instance=data)
print(response)

```

# Deleting Data

To delete data from QuipuBase, use the `delete` method:

```python

key = "your_data_key"
response = await client.delete(namespace="my_namespace", key=key)
print(response)

```

## Finding Data

To find data based on certain criteria, use the `find` method:

```python

response = await client.find(namespace="my_namespace", name="example")
print(response)

```

# Advanced Operations

## Upserting Vector Data

QuipuBase supports vector operations for advanced use cases such as similarity search and AI applications:

```python

from quipubase_client.schemas import RagRequest

data = RagRequest(content=[0.1, 0.2, 0.3, 0.4])
response = await client.upsert(namespace="my_namespace", data=data)
print(response)

```

## Querying Vector Data

To query vector data:

```python

response = await client.query(namespace="my_namespace", data=data, top_k=5)
print(response)
# CosimResult(id='23rw-a194k-2r3', score=0.99, content='example')

```


# Conclusion

The QuipuBase Client for Python provides a powerful and efficient way to interact with QuipuBase, enabling developers to build advanced AI applications with ease. With high-performance storage, semantic search, custom data modeling, and seamless integrations, QuipuBase is the ideal choice for modern data management needs. Get started with the QuipuBase Client for Python today and unlock the full potential of your applications!

For more information, visit [QuipuBase](https://www.quipubase.online).

