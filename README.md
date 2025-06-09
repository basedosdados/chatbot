# Chatbot

**Chatbot** is a Python library designed to make it easy for large language models (LLMs) to interact with your data. It is built on top of [LangChain](https://python.langchain.com/docs/introduction/) and [LangGraph](https://langchain-ai.github.io/langgraph/concepts/why-langgraph/) and provides agents and high-level assistants for natural language querying and data visualization.

> [!NOTE]
> This library is **still under active development**. Expect breaking changes, incomplete features, and limited documentation.

## Installation
Clone the repository and install it (you can also use [poetry](https://python-poetry.org/) or [uv](https://docs.astral.sh/uv/) instead of pip).
```bash
git clone https://github.com/basedosdados/chatbot.git
cd chatbot
pip install .
```

## Assistants

### SQLAssistant
The `SQLAssistant` allows LLMs to interact with your database so you can ask questions about it. All it needs is a LangChain [ChatModel](https://python.langchain.com/docs/integrations/chat/) and a [Database](https://github.com/basedosdados/chatbot/blob/fc1269826229e4daad5c6cc7678ab55dc4739c08/chatbot/databases/database.py)-like object:
```python
import os

from langchain.chat_models import init_chat_model

from chatbot.assistants import SQLAssistant
from chatbot.databases import BigQueryDatabase

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path to your google cloud service account"

db = BigQueryDatabase(
    billing_project="your billing project", # you could also set the "BILLING_PROJECT_ID" env variable
    query_project="your query project", # you could also set the "QUERY_PROJECT_ID" env variable
)

model = init_chat_model("gpt-4o", temperature=0)

assistant = SQLAssistant(db, model)

response = assistant.invoke("hello! what can you tell me about our database?")
```

You can optionally use a [PostgresSaver](https://langchain-ai.github.io/langgraph/reference/checkpoints/#langgraph.checkpoint.postgres.PostgresSaver) checkpointer to add short-term memory to your assistant and a [VectorStore](https://python.langchain.com/docs/integrations/vectorstores/) for few-shot prompting during SQL queries generation:
```python
import os

from langchain.chat_models import init_chat_model

from langchain_chroma import Chroma
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

from chatbot.assistants import SQLAssistant
from chatbot.databases import BigQueryDatabase

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path to your google cloud service account"

db = BigQueryDatabase(
    billing_project="your billing project", # you could also set the "BILLING_PROJECT_ID" env variable
    query_project="your query project", # you could also set the "QUERY_PROJECT_ID" env variable
)

model = init_chat_model("gpt-4o", temperature=0)

# it could be any vector store
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=OpenAIEmbeddings(
      model="text-embedding-3-small",
    ),
)

DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres

with PostgresSaver.from_conn_strin(DB_URI) as checkpointer:
    checkpointer.setup()

    assistant = SQLAssistant(
        database=db,
        model=model,
        checkpointer=checkpointer,
        vector_store=vector_store,
    )

    response = assistant.invoke(
        message="hello! what can you tell me about our database?",
        thread_id="uuid"
    )
```

An async version is also available: `AsyncSQLAssistant`.

### SQLVizAssistant
`SQLVizAssistant` extends `SQLAssistant` by not only retrieving data but also **preparing it for visualization**. It identifies which variables should be plotted on each axis, suggests appropriate chart types, and defines metadata such as titles, labels, and legends, without rendering the chart itself. Again, all it needs is a LangChain [ChatModel](https://python.langchain.com/docs/integrations/chat/) and a [Database](https://github.com/basedosdados/chatbot/blob/fc1269826229e4daad5c6cc7678ab55dc4739c08/chatbot/databases/database.py)-like object:
```python
import os

from langchain.chat_models import init_chat_model

from chatbot.assistants import SQLVizAssistant
from chatbot.databases import BigQueryDatabase

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path to your google cloud service account"

db = BigQueryDatabase(
    billing_project="your billing project", # you could also set the "BILLING_PROJECT_ID" env variable
    query_project="your query project", # you could also set the "QUERY_PROJECT_ID" env variable
)

model = init_chat_model("gpt-4o", temperature=0)

assistant = SQLVizAssistant(db, model)

response = assistant.invoke("hello! what can you tell me about our database?")
```

You can also optionally use a PostgresSaver checkpointer to add short-term memory to your assistant, and provide separate vector stores for few-shot prompting in both SQL generation and visualization reasoning:
```python
import os

from langchain.chat_models import init_chat_model

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

# it could be any vector store
sql_vector_store = Chroma(
    collection_name="example_sql_collection",
    embedding_function=OpenAIEmbeddings(
      model="text-embedding-3-small",
    ),
)

# it could also be any vector store!
viz_vector_store = Chroma(
    collection_name="example_viz_collection",
    embedding_function=OpenAIEmbeddings(
      model="text-embedding-3-small",
    ),
)

DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()

    assistant = SQLVizAssistant(
        database=db,
        model=model,
        checkpointer=checkpointer,
        sql_vector_store=sql_vector_store,
        viz_vector_store=viz_vector_store
    )

    response = assistant.invoke(
        message="hello! what can you tell me about our database?",
        thread_id="uuid"
    )
```
An async version is also available: `AsyncSQLVizAssistant`.

## Extensibility
Under the hood, both assistants rely on composable agents:

- `SQLAgent` – Handles database metadata retrieval, query generation and execution.
- `VizAgent` – Handles visualization reasoning.
- `RouterAgent` – Routes between `SQLAgent` and `VizAgent` depending on the user message.

You can directly use these agents or use them to create your own workflows.
