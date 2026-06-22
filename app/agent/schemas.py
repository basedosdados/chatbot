from enum import Enum

from pydantic import BaseModel, Field


class TemporalGranularity(str, Enum):
    """Granularidade do período de cobertura dos dados utilizados."""

    DAY = "day"
    MONTH = "month"
    YEAR = "year"


class TemporalCoverage(BaseModel):
    """Intervalo efetivamente filtrado pela consulta SQL na resposta."""

    period_start: str = Field(
        description=(
            "Início do intervalo filtrado pela consulta SQL (ex.: '2010' para "
            "`ano = 2010` ou `ano BETWEEN 2010 AND 2012`). Pode ser mais estreito "
            "que a cobertura total da tabela."
        )
    )
    period_end: str = Field(
        description=(
            "Fim do intervalo filtrado pela consulta SQL (ex.: '2010' para "
            "`ano = 2010`; '2012' para `ano BETWEEN 2010 AND 2012`). Pode ser mais "
            "estreito que a cobertura total da tabela."
        )
    )
    granularity: TemporalGranularity = Field(
        description="Granularidade de `period_start`/`period_end`: 'day', 'month' ou 'year'."
    )


class DataSource(BaseModel):
    """Uma tabela da Base dos Dados utilizada para responder à pergunta."""

    dataset_id: str = Field(
        description=(
            "UUID do dataset (campo `dataset_id` de `get_table_details` ou `id` de "
            "`get_dataset_details`), não o nome BigQuery (ex.: 'br_bd_diretorios')."
        )
    )
    table_id: str = Field(
        description=(
            "UUID da tabela (campo `id` de `get_table_details`), não o nome BigQuery."
        )
    )
    name: str = Field(description="Nome legível da tabela utilizada.")


class StructuredResponse(BaseModel):
    """Resposta estruturada do agente para a interface do usuário."""

    response: str = Field(
        description=(
            "A resposta em prosa (Markdown): resposta direta à pergunta com os dados "
            "obtidos, mais análise e contexto. Não repita aqui fonte, período, SQL "
            "ou sugestões — cada um desses elementos tem seu campo dedicado."
        )
    )
    data_sources: list[DataSource] | None = Field(
        default=None,
        description=(
            "As tabelas efetivamente consultadas para responder à pergunta. Deixe "
            "vazio (None) quando a resposta não usar dados de tabelas (ex.: explicar "
            "a plataforma, pedir esclarecimento ou listar os tipos de dados disponíveis)."
        ),
    )
    temporal_coverage: TemporalCoverage | None = Field(
        default=None,
        description=(
            "O período geral dos dados utilizados. Deixe vazio (None) "
            "quando a resposta não envolver uma dimensão temporal."
        ),
    )
    sql_query: str | None = Field(
        default=None,
        description=(
            "A consulta SQL executada para obter os dados, com comentários inline. "
            "Deixe vazio (None) quando nenhuma consulta foi executada."
        ),
    )
    follow_up_questions: list[str] = Field(
        default_factory=list,
        description="3 sugestões de perguntas para explorar os dados mais a fundo.",
    )
