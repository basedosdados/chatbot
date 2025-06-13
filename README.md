# Chatbot

**Chatbot** is a Python library designed to make it easy for Large Language Models (LLMs) to interact with your data. It is built on top of [LangChain](https://python.langchain.com/docs/introduction/) and [LangGraph](https://langchain-ai.github.io/langgraph/concepts/why-langgraph/) and provides agents and high-level assistants for natural language querying and data visualization.

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
The [`SQLAssistant`](https://github.com/basedosdados/chatbot/blob/d5a1c275183932de52781af6346d06b1c148e675/chatbot/assistants/sql_assistant.py) allows LLMs to interact with your database so you can ask questions about it. All it needs is a LangChain [Chat Model](https://python.langchain.com/docs/integrations/chat/), a [Context Provider](https://github.com/basedosdados/chatbot/blob/d5a1c275183932de52781af6346d06b1c148e675/chatbot/contexts/context_provider.py) and a [Prompt Formatter](https://github.com/basedosdados/chatbot/blob/d5a1c275183932de52781af6346d06b1c148e675/chatbot/formatters/prompt_formatter.py). The context provider is responsible for providing context about your data to the SQL Agent and the prompt formatter is responsible for building a system prompt for **SQL generation**.

We provide a default [`BigQueryContextProvider`](https://github.com/basedosdados/chatbot/blob/d5a1c275183932de52781af6346d06b1c148e675/chatbot/contexts/bigquery_context_provider.py) for retrieving metadata directly from Google BigQuery and a default [`SQLPromptFormatter`](https://github.com/basedosdados/chatbot/blob/d5a1c275183932de52781af6346d06b1c148e675/chatbot/formatters/sql_prompt_formatter.py). You can supply your own implementation of a context provider and a prompt formatter for custom behaviour.
```python
from langchain.chat_models import init_chat_model

from chatbot.assistants import SQLAssistant
from chatbot.contexts import BigQueryContextProvider
from chatbot.formatters import SQLPromptFormatter

model = init_chat_model("gpt-4o", temperature=0)

# you must point the GOOGLE_APPLICATION_CREDENTIALS
# env variable to your service account JSON file.
context_provider = BigQueryContextProvider(
    billing_project="your billing project",
    query_project="your query project",
)

prompt_formatter = SQLPromptFormatter()

assistant = SQLAssistant(model, context_provider, prompt_formatter)

response = assistant.invoke("hello! what can you tell me about our database?")
```

You can optionally use a [`PostgresSaver`](https://langchain-ai.github.io/langgraph/reference/checkpoints/#langgraph.checkpoint.postgres.PostgresSaver) checkpointer to add short-term memory to your assistant and a [`VectorStore`](https://python.langchain.com/docs/integrations/vectorstores/) for few-shot prompting during **SQL queries generation**:
```python
from langchain.chat_models import init_chat_model

from langchain_postgres import PGVector
from langgraph.checkpoint.postgres import PostgresSaver

from chatbot.assistants import SQLAssistant
from chatbot.contexts import BigQueryContextProvider
from chatbot.formatters import SQLPromptFormatter

model = init_chat_model("gpt-4o", temperature=0)

# you must point the GOOGLE_APPLICATION_CREDENTIALS
# env variable to your service account JSON file.
context_provider = BigQueryContextProvider(
    billing_project="your billing project",
    query_project="your query project",
)

# it could be any combination of
# a langchain vector store and an embedding model
vector_store = PGVector(
    connection="your connection string",
    collection_name="your collection name",
    embedding=OpenAIEmbeddings(
      model="text-embedding-3-small",
    ),
)

prompt_formatter = SQLPromptFormatter(vector_store)

DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres

with PostgresSaver.from_conn_strin(DB_URI) as checkpointer:
    checkpointer.setup()

    assistant = SQLAssistant(
        model=model,
        context_provider=context_provider,
        prompt_formatter=prompt_formatter,
        checkpointer=checkpointer,
    )

    response = assistant.invoke(
        message="hello! what can you tell me about our database?",
        thread_id="some uuid"
    )
```

An async version is also available: [`AsyncSQLAssistant`](https://github.com/basedosdados/chatbot/blob/d5a1c275183932de52781af6346d06b1c148e675/chatbot/assistants/async_sql_assistant.py).

### SQLVizAssistant
[`SQLVizAssistant`](https://github.com/basedosdados/chatbot/blob/d5a1c275183932de52781af6346d06b1c148e675/chatbot/assistants/sql_viz_assistant.py) extends `SQLAssistant` by not only retrieving data but also **preparing it for visualization**. It identifies which variables should be plotted on each axis, suggests appropriate chart types, and defines metadata such as titles, labels, and legends, without plotting the chart itself. It needs a LangChain Chat Model, a Context Provider and two separate Prompt Formatters: One for **SQL generation** and the other for guiding **data preprocessing for visualization**.

We provide a default [`VizPromptFormatter`](https://github.com/basedosdados/chatbot/blob/d5a1c275183932de52781af6346d06b1c148e675/chatbot/formatters/viz_prompt_formatter.py), which is used internally by the Visualization Agent during **data preprocessing**.
```python
from langchain.chat_models import init_chat_model

from chatbot.assistants import SQLAssistant
from chatbot.contexts import BigQueryContextProvider
from chatbot.formatters import SQLPromptFormatter, VizPromptFormatter

model = init_chat_model("gpt-4o", temperature=0)

# you must point the GOOGLE_APPLICATION_CREDENTIALS
# env variable to your service account JSON file.
context_provider = BigQueryContextProvider(
    billing_project="your billing project",
    query_project="your query project",
)

sql_prompt_formatter = SQLPromptFormatter()
viz_prompt_formatter = VizPromptFormatter()

assistant = SQLVizAssistant(
  model, context_provider, sql_prompt_formatter, viz_prompt_formatter
)

response = assistant.invoke("hello! what can you tell me about our database?")
```

You can also optionally use a `PostgresSaver` checkpointer to add short-term memory to your assistant, and provide langchain vector stores for few-shot prompting during both **SQL generation** and **data preprocessing for visualization**:
```python
from langchain.chat_models import init_chat_model

from langchain_postgres import PGVector
from langgraph.checkpoint.postgres import PostgresSaver

from chatbot.assistants import SQLAssistant
from chatbot.contexts import BigQueryContextProvider
from chatbot.formatters import SQLPromptFormatter, VizPromptFormatter

model = init_chat_model("gpt-4o", temperature=0)

# you must point the GOOGLE_APPLICATION_CREDENTIALS
# env variable to your service account JSON file.
context_provider = BigQueryContextProvider(
    billing_project="your billing project",
    query_project="your query project",
)

# it could be any combination of
# a langchain vector store and an embedding model
sql_vector_store = PGVector(
    connection="your connection string",
    collection_name="your sql collection name",
    embedding=OpenAIEmbeddings(
      model="text-embedding-3-small",
    ),
)

viz_vector_store = PGVector(
    connection="your connection string",
    collection_name="your viz collection name",
    embedding=OpenAIEmbeddings(
      model="text-embedding-3-small",
    ),
)

sql_prompt_formatter = SQLPromptFormatter(sql_vector_store)
viz_prompt_formatter = VizPromptFormatter(viz_vector_store)

DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres

with PostgresSaver.from_conn_strin(DB_URI) as checkpointer:
    checkpointer.setup()

    assistant = SQLAssistant(
        model=model,
        context_provider=context_provider,
        sql_prompt_formatter=sql_prompt_formatter,
        viz_prompt_formatter=viz_prompt_formatter,
        checkpointer=checkpointer,
    )

    response = assistant.invoke(
        message="hello! what can you tell me about our database?",
        thread_id="some uuid"
    )
```
An async version is also available: [`AsyncSQLVizAssistant`](https://github.com/basedosdados/chatbot/blob/d5a1c275183932de52781af6346d06b1c148e675/chatbot/assistants/async_sql_viz_assistant.py).

## Extensibility
Under the hood, both assistants rely on composable agents:

- [`SQLAgent`](https://github.com/basedosdados/chatbot/blob/d5a1c275183932de52781af6346d06b1c148e675/chatbot/agents/sql_agent.py) – Handles database metadata retrieval, query generation and execution.
- [`VizAgent`](https://github.com/basedosdados/chatbot/blob/d5a1c275183932de52781af6346d06b1c148e675/chatbot/agents/visualization_agent.py) – Handles visualization reasoning.
- [`RouterAgent`](https://github.com/basedosdados/chatbot/blob/d5a1c275183932de52781af6346d06b1c148e675/chatbot/agents/router_agent.py) – Orchestrates SQL querying and data visualization via a multi-agent workflow..

There is also an implementation of a simple [`ReActAgent`](https://github.com/basedosdados/chatbot/blob/d5a1c275183932de52781af6346d06b1c148e675/chatbot/agents/react_agent.py) with support to custom system prompts and short-term memory, to which you can add an arbitrary set of tools.

You can directly use these agents or use them to create your own workflows.
