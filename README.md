<div align="center">
    <a href="https://basedosdados.org">
        <img src="https://storage.googleapis.com/basedosdados-website/logos/bd_minilogo.png" width="240" alt="Base dos Dados">
    </a>
</div>

# Chatbot da Base dos Dados
API do chatbot da [Base dos Dados](https://basedosdados.org), a maior plataforma pública de dados do Brasil.

## Tech Stack
- **[FastAPI](https://fastapi.tiangolo.com/)** como framework web assíncrono.
- **[Pydantic](https://docs.pydantic.dev/latest/)** para validação de dados e gerenciamento de configurações.
- **[SQLModel](https://sqlmodel.tiangolo.com/)** para interação com o banco de dados (ORM).
- **[PostgreSQL](https://www.postgresql.org/)** como banco de dados relacional.
- **[Alembic](https://alembic.sqlalchemy.org/en/latest/)** para migrações.
- **[LangChain](https://docs.langchain.com/oss/python/langchain/overview)** para construção de agentes de IA.
- **[Google BigQuery](https://cloud.google.com/bigquery)** como fonte dos dados tratados.
- **[Google VertexAI](https://cloud.google.com/vertex-ai)** como provedor de LLMs.
- **[Docker](https://www.docker.com/)** para conteinerização e desenvolvimento local.
- **[Helm](https://helm.sh/pt/)** e **[Google Kubernetes Engine](https://cloud.google.com/kubernetes-engine)** para deploy em produção.
- **[GitHub Actions](https://github.com/features/actions)** para automação de fluxos de trabalho de CI/CD.
- **[pre-commit](https://pre-commit.com/)** para gerenciamento de hooks de pre-commit.
- **[ruff](https://docs.astral.sh/ruff/)** para linting e formatação.
- **[uv](https://docs.astral.sh/uv/)** para gerenciamento de dependências.

## Configuração do Ambiente de Desenvolvimento
Para funcionar adequadamente, a API do chatbot depende da API do website da Base dos Dados, com a qual compartilha o banco de dados. Siga as instruções abaixo para executá-las na ordem correta.

### 1. Configuração da API do website
Instale o [Docker](https://docs.docker.com/engine/install/). Em seguida, clone o repositório do [backend](https://github.com/basedosdados/backend):
```bash
git clone https://github.com/basedosdados/backend.git
cd backend
```

Configure o ambiente de acordo com as [instruções do repositório](https://github.com/basedosdados/backend?tab=readme-ov-file#configura%C3%A7%C3%A3o-do-ambiente-de-desenvolvimento) e execute utilizando o docker compose:
```bash
docker compose up
```

> [!IMPORTANT]
> O backend do website deve ser executado **antes** da API do chatbot.

> [!TIP]
> Caso deseje, você pode executar o backend em segundo plano:
> ```bash
> docker compose up -d
> ```

### 2. Configuração da API do chatbot
Instale o [uv](https://docs.astral.sh/uv/getting-started/installation/):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone o repositório do chatbot:
```bash
git clone https://github.com/basedosdados/chatbot.git
cd chatbot
```

Crie um ambiente virtual:
```bash
uv sync
```

Instale os hooks de pre-commit:
```bash
pre-commit install
```

Copie o arquivo `.env.example` e configure as variáveis de ambiente:
```bash
cp .env.example .env
```

> [!IMPORTANT]
> As seguintes variáveis devem ser **idênticas** entre as APIs:
>
> | API Chatbot | API Website |
> |-------------|-------------|
> | `DB_HOST` | `DB_HOST` |
> | `DB_PORT` | `DB_PORT` |
> | `DB_USER` | `DB_USER` |
> | `DB_PASSWORD` | `DB_PASSWORD` |
> | `DB_NAME` | `DB_NAME` |
> | `JWT_ALGORITHM` | `DJANGO_JWT_ALGORITHM` |
> | `JWT_SECRET_KEY` | `DJANGO_SECRET_KEY` |
>
> Além disso, você precisará de uma conta de serviço com acesso ao BigQuery e à VertexAI, chamada `chatbot-sa.json` e armazenada em `${HOME}/.basedosdados/credentials`.

### 3. Executando a API
**Com o [Compose Watch](https://docs.docker.com/compose/how-tos/file-watch/) (recomendado):**
```bash
docker compose up --watch
```

**Manualmente com o uv:**
```
uv run alembic upgrade head
uv run fastapi dev --host 0.0.0.0 app/main.py
```

> [!NOTE]
> Caso opte por executar manualmente, altere `DB_HOST` para `localhost` e `GOOGLE_SERVICE_ACCOUNT` para `${HOME}/.basedosdados/credentials/chatbot-sa.json`.
