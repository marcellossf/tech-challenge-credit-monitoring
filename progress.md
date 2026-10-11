# Progresso — 10/10/2026

- Pasta inicialmente só com .git, sem commits nem remote.
- Habilidades PDF e planning-with-files lidas; consulta breve à preferência acadêmica anterior.
- PDF extraído com pypdf/renderizado com pypdfium2; rubrica conferida visualmente.
- Falta de fitz e saída cp1252 resolvidas com bibliotecas disponíveis/arquivo UTF-8.
- Usuário confirmou prazo/individual/vídeo até 5min e pediu repo.
- Subagente revisou requisitos em paralelo, sem editar arquivos.
- Fontes UCI/Evidently/LGPD verificadas.
- Patch inicial rejeitado por operações duplicadas, sem alteração parcial; refeito por gravação direta.

- Repositório privado criado: https://github.com/marcellossf/tech-challenge-credit-monitoring. Planejamento inicial preparado para um único commit real; implementação futura por etapas.

## Implementação iniciada — 10/10/2026

- Repo alterado para PUBLIC e verificado com GitHub CLI.
- Ambiente .venv criado com Python 3.13.6; instalação isolada em andamento.
- Contratos, simulações, estatísticas, monitoramento e HTML implementados em paralelo.
- Root implementou download oficial, split, baseline, fronteira de inferência e CLI integrado.
- Testes de agentes em ambiente global preliminares; aceite final exige .venv e demo real.

- .venv concluída: Python 3.13.6, Pandera 0.34.1, Evidently 0.7.23, MLflow 3.17.0, sklearn 1.9.1. Lock de 142 dependências, pywin32 com marcador Windows.
- Download real UCI: 30.000 linhas; 18.000/6.000/6.000; hash original fixado para rejeitar mudança não revisada. Nenhuma linha rejeitada na preparação.
- 38 testes passaram no ambiente isolado; pip check e Ruff passaram. Warnings são deprecations internas de dependências.
- Demo completa passou nas 12 verificações: controle saudável, inválido bloqueado, duas features com drift e conceito com X preservado/alerta de desempenho.
- HTML conferido visualmente em Chrome 1440x1000; erro inicial somente favicon inexistente, corrigido com ícone inline. CLI Playwright bloqueia file://; servido apenas reports em localhost.
- Revisão por subagente: paths de relatórios normalizados e run.json inclui MLflow ID; código recebe fingerprint para rastreabilidade.

- Código publicado no commit 40fca71; demo final regenerada contra esse commit, batch 799668cec6b74677ba96a7a814dc71b2.
- GitHub Actions 38099510133: SUCCESS em Linux, instalação lock + pip check + Ruff + 38 testes + demo completa + upload de evidências.
- Interface MLflow validada em Model training/credit-monitoring; Windows exige --workers 1, porta 5001 usada. Tentativa inicial com múltiplos workers falhou WinError10022 e porta5000 residual, processos da tentativa própria encerrados.
- Reports e docs revisados: sem caminhos do usuário ou credenciais; gráficos Evidently agregados e sem ID/labels.
- Vídeo e submissão continuam pendentes do autor.
