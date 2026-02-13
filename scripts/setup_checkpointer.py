"""
Run LangGraph checkpointer setup. `checkpointer.setup()` should be called only once.
Ref: https://docs.langchain.com/oss/python/langgraph/add-memory#example-using-postgres-checkpointer
"""

from langgraph.checkpoint.postgres import PostgresSaver

from app.settings import settings


def main():
    with PostgresSaver.from_conn_string(settings.DB_URL) as checkpointer:
        checkpointer.setup()


if __name__ == "__main__":
    main()
