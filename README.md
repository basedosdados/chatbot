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

### 1. Pré-requisitos
Instale o [Docker](https://docs.docker.com/engine/install/) e o [uv](https://docs.astral.sh/uv/getting-started/installation/). Em seguida, clone o repositório do chatbot:
```bash
git clone https://github.com/basedosdados/chatbot.git
cd chatbot
```

Crie um ambiente virtual e instale os hooks de pre-commit:
```bash
uv sync
pre-commit install
```

Copie o arquivo `.env.example` e configure as variáveis de ambiente:
```bash
cp .env.example .env
```

> [!IMPORTANT]
> Você precisará de uma conta de serviço com acesso ao BigQuery e à VertexAI, chamada `chatbot-sa.json` e armazenada em `${HOME}/.basedosdados/credentials`.

### 2. Modo de desenvolvimento independente
Por padrão, a API do chatbot exige um token JWT válido emitido pela API do website. Para desenvolvimento local sem depender da API do website, habilite o modo de autenticação de desenvolvedor no arquivo `.env`:
```bash
AUTH_DEV_MODE=true
AUTH_DEV_USER_ID=1
```
> [!NOTE]
> O modo de autenticação de desenvolvedor só funciona quando `ENVIRONMENT=development`.

> [!WARNING]
> O modo de autenticação de desenvolvedor ignora a validação do token JWT e retorna o ID definido por `AUTH_DEV_USER_ID` para todas as requisições. **Nunca habilite em produção.**

### 3. Executando a API
**Com o [Compose Watch](https://docs.docker.com/compose/how-tos/file-watch/) (recomendado):**
```bash
docker compose up --watch
```

**Manualmente com o uv:**
```bash
uv run alembic upgrade head
uv run fastapi dev --host 0.0.0.0 app/main.py
```
> [!NOTE]
> Caso opte por executar a API manualmente, você precisará configurar uma instância do PostgreSQL ou executar o serviço `database` do compose file com `docker compose up database`.
> Em ambos os casos,<br>ajuste as variáveis `DB_*` no `.env` conforme necessário para conectar-se ao banco.
>
> Além disso, aponte a variável `GOOGLE_SERVICE_ACCOUNT` para o caminho local da conta de serviço.

## Executando a Aplicação Completa (Full Stack)
Para testar a integração completa com o frontend e a API do website, siga as instruções abaixo.

### 1. Configuração da API do website
Clone o repositório do [backend](https://github.com/basedosdados/backend):
```bash
git clone https://github.com/basedosdados/backend.git
cd backend
```

Configure e execute de acordo com as [instruções do repositório](https://github.com/basedosdados/backend?tab=readme-ov-file#configura%C3%A7%C3%A3o-do-ambiente-de-desenvolvimento).

### 2. Configuração da API do chatbot
Desabilite o modo de autenticação de desenvolvedor e configure as variáveis `JWT_*` no arquivo `.env` da API do chatbot:
```bash
AUTH_DEV_MODE=false
JWT_ALGORITHM=jwt-algorithm
JWT_SECRET_KEY=jwt-secret-key
```

> [!IMPORTANT]
> As seguintes variáveis devem ser **idênticas** entre as APIs:
>
> | API Chatbot | API Website |
> |-------------|-------------|
> | `JWT_ALGORITHM` | `DJANGO_JWT_ALGORITHM` |
> | `JWT_SECRET_KEY` | `DJANGO_SECRET_KEY` |

Execute a API do chatbot:
```bash
docker compose up --watch
```

### 3. Configuração do frontend
Clone o repositório [chatbot-frontend](https://github.com/basedosdados/chatbot-frontend):
```bash
git clone git@github.com:basedosdados/chatbot-frontend.git
cd chatbot-frontend
```

Configure e execute de acordo com as [instruções do repositório](https://github.com/basedosdados/chatbot-frontend?tab=readme-ov-file#interface-do-chatbot-da-bd-feita-com-streamlit).
