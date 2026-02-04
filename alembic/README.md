# Migração do Banco de Dados

1\. Configure as variáveis de ambiente `DB_*` para conectar-se ao banco de dados.

2\. Gere o script de migração:
```bash
alembic revision --autogenerate -m "alguma mensagem descritiva"
```

> [!WARNING]
> Revise cuidadosamente o script de migração antes de executá-lo. Ajustes manuais podem ser necessários para garantir que ele funcione corretamente.

3\. Execute o script de migração:
```bash
alembic upgrade head
```

> [!WARNING]
> Teste o script de migração em ambientes de desenvolvimento e de staging antes de executá-lo em produção.
