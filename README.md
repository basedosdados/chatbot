# Chatbot

**Chatbot** is a Python library designed to make it easy for large language models (LLMs) to interact with your data. It is built on top of [LangChain](https://python.langchain.com/docs/introduction/) and [LangGraph](https://langchain-ai.github.io/langgraph/concepts/why-langgraph/) and provides agents and high-level assistants for natural language querying and data visualization.

> [!NOTE]
> This library is **still under active development**. Expect breaking changes, incomplete features, and limited documentation.

## Installation
Clone the repository and install it using pip (you can also use [poetry](https://python-poetry.org/) or [uv](https://docs.astral.sh/uv/)).
```bash
git clone https://github.com/basedosdados/chatbot.git
cd chatbot
pip install .
```

## Assistants

### SQLAssistant
The `SQLAssistant` allows LLMs to interact with your database so you can ask questions about it. All it needs is a [Database](https://github.com/basedosdados/chatbot/blob/fc1269826229e4daad5c6cc7678ab55dc4739c08/chatbot/databases/database.py) object:
```python
import os

from chatbot.assistants import SQLAssistant, UserMessage
from chatbot.databases import BigQueryDatabase

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path to your google cloud service account"

db = BigQueryDatabase(
    billing_project="your billing project", # you could also set the "BILLING_PROJECT_ID" env variable
    query_project="your query project", # you could also set the "QUERY_PROJECT_ID" env variable
)

assistant = SQLAssistant(db, model_uri="openai/gpt-4o") # you could also set the "MODEL_URI" env variable

message = UserMessage(content="your question")

response = assistant.invoke(message)
```

Optionally, you can use a [PostgresSaver](https://langchain-ai.github.io/langgraph/reference/checkpoints/#langgraph.checkpoint.postgres.PostgresSaver) checkpointer to add persistence to your assistant and a [VectorStore](https://python.langchain.com/docs/integrations/vectorstores/) for few-shot prompting during SQL queries generation:
```python
import os

from langchain_chroma import Chroma
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

from chatbot.assistants import SQLAssistant, UserMessage
from chatbot.databases import BigQueryDatabase

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path to your google cloud service account"

db = BigQueryDatabase(
    billing_project="your billing project", # you could also set the "BILLING_PROJECT_ID" env variable
    query_project="your query project", # you could also set the "QUERY_PROJECT_ID" env variable
)

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=OpenAIEmbeddings(
      model="text-embedding-3-small",
    ),
)

conn_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0
}

with ConnectionPool(
    conninfo=db_url,
    max_size=8,
    kwargs=conn_kwargs
) as pool:
    checkpointer = PostgresSaver(pool)
    checkpointer.setup()

    assistant = SQLAssistant(
        database=db,
        model_uri="openai/gpt-4o", # you could also set the "MODEL_URI" env variable
        checkpointer=checkpointer,
        vector_store=vector_store,
    )

    message = UserMessage(content="your question")

    response = assistant.invoke(message, thread_id="your_thread_id")
```

An async version is also available: `AsyncSQLAssistant`.

### SQLVizAssistant
`SQLVizAssistant` extends `SQLAssistant`, by not only retrieving data but also preparing it for visualization. It identifies which variables should be plotted on each axis, suggests appropriate chart types, and defines metadata such as titles, labels, and legends, without rendering the chart itself. Again, all it needs is a [Database](https://github.com/basedosdados/chatbot/blob/fc1269826229e4daad5c6cc7678ab55dc4739c08/chatbot/databases/database.py) object:
```python
import os

from chatbot.assistants import SQLVizAssistant, UserMessage
from chatbot.databases import BigQueryDatabase

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path to your google cloud service account"

db = BigQueryDatabase(
    billing_project="your billing project", # you could also set the "BILLING_PROJECT_ID" env variable
    query_project="your query project", # you could also set the "QUERY_PROJECT_ID" env variable
)

assistant = SQLVizAssistant(db, model_uri="openai/gpt-4o") # you could also set the "MODEL_URI" env variable

message = UserMessage(content="your question")

response = assistant.invoke(message)
```

You can also optionally use a PostgresSaver checkpointer and provide separate vector stores for few-shot prompting on both SQL generation and visualization reasoning:
```python
import os

from langchain_chroma import Chroma
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

from chatbot.assistants import SQLVizAssistant, UserMessage
from chatbot.databases import BigQueryDatabase

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path to your google cloud service account"

db = BigQueryDatabase(
    billing_project="your billing project", # you could also set the "BILLING_PROJECT_ID" env variable
    query_project="your query project", # you could also set the "QUERY_PROJECT_ID" env variable
)

sql_vector_store = Chroma(
    collection_name="example_sql_collection",
    embedding_function=OpenAIEmbeddings(
      model="text-embedding-3-small",
    ),
)

viz_vector_store = Chroma(
    collection_name="example_viz_collection",
    embedding_function=OpenAIEmbeddings(
      model="text-embedding-3-small",
    ),
)

conn_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0
}

with ConnectionPool(
    conninfo=db_url,
    max_size=8,
    kwargs=conn_kwargs
) as pool:
    checkpointer = PostgresSaver(pool)
    checkpointer.setup()

    assistant = SQLVizAssistant(
        database=db,
        model_uri="openai/gpt-4o", # you could also set the "MODEL_URI" env variable
        checkpointer=checkpointer,
        sql_vector_store=sql_vector_store,
        viz_vector_store=viz_vector_store
    )

    message = UserMessage(content="your question")

    response = assistant.invoke(message, thread_id="your_thread_id")
```
An async version is also available: `AsyncSQLVizAssistant`.

## Extensibility
Under the hood, both assistants rely on composable agents:

- `SQLAgent` – Handles database metadata retrieval, query generation and execution.
- `VizAgent` – Handles visualization reasoning.
- `RouterAgent` – Routes between SQLAgent and VizAgent depending on the user message.

You can directly use these agents or use them to create your own workflows.
