from enum import Enum

from pydantic import BaseModel, Field


class TemporalGranularity(str, Enum):
    """Granularidade do período de cobertura dos dados utilizados."""

    DAY = "day"
    MONTH = "month"
    YEAR = "year"


class TemporalCoverage(BaseModel):
    """Período de cobertura dos dados utilizados na resposta."""

    period_start: str = Field(
        description=(
            "Início do período **efetivamente consultado** na consulta SQL (ex.: se a "
            "consulta filtrou `ano = 2010`, use '2010'; se filtrou `ano BETWEEN 2010 AND "
            "2012`, use '2010'). NÃO use o `period_start` dos metadados da tabela — "
            "use o valor que aparece nos filtros temporais da consulta SQL executada."
        )
    )
    period_end: str = Field(
        description=(
            "Fim do período **efetivamente consultado** na consulta SQL (ex.: se a consulta "
            "filtrou `ano = 2010`, use '2010'; se filtrou `ano BETWEEN 2010 AND 2012`, "
            "use '2012'). NÃO use o `period_end` dos metadados da tabela — use o "
            "valor que aparece nos filtros temporais da consulta SQL executada."
        )
    )
    granularity: TemporalGranularity = Field(
        description=(
            "Granularidade do período: 'day' (dia), 'month' (mês) ou 'year' (ano), "
            "de acordo com o formato de `period_start`/`period_end`."
        )
    )


class DataSource(BaseModel):
    """Uma tabela da Base dos Dados utilizada para responder à pergunta."""

    dataset_id: str = Field(
        description=(
            "UUID do dataset retornado pelo campo `dataset_id` de `get_table_details` "
            "ou pelo campo `id` de `get_dataset_details` (ex.: 'a1b2c3d4-...'). "
            "NÃO use o `gcp_id` nem o nome BigQuery do dataset "
            "(ex.: 'br_bd_diretorios') — use o UUID da API da Base dos Dados."
        )
    )
    table_id: str = Field(
        description=(
            "UUID da tabela retornado pelo campo `id` de `get_table_details` (ex.: '3027c0d8-...'). "
            "NÃO use o `gcp_id` nem o nome BigQuery da tabela — use o UUID da API da Base dos Dados."
        )
    )
    name: str = Field(description="Nome legível da tabela utilizada.")


class StructuredResponse(BaseModel):
    """Resposta estruturada do agente para a interface do usuário."""

    response: str = Field(
        description=(
            "A resposta em prosa exibida ao usuário, em Markdown. Contém a resposta "
            "direta à pergunta com os dados obtidos, além de análise e contexto "
            "relevante. NÃO inclua aqui a fonte/tabelas, o período de cobertura, a "
            "consulta SQL nem as sugestões de exploração — esses elementos são "
            "retornados nos campos dedicados."
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
