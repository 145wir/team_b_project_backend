# TEAM B Backend

Team B AI Paper Agent의 FastAPI Backend입니다.

현재 단계에서는 AWS RDS PostgreSQL에 저장된
`core_documents` 데이터를 조회하는 기능부터 구현합니다.

---

## Architecture

현재 전체 프로젝트 역할은 다음과 같습니다.

```text
Database Project
arXiv / NASA NTRS
        ↓
Collector
        ↓
Resolver
        ↓
Parser
        ↓
Cleaner
        ↓
Loader
        ↓
AWS RDS PostgreSQL
        ↓
Backend
        ↓
RAG
        ↓
Own Transformer
        ↓
Frontend