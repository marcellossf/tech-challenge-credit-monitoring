# Evidências e fontes

Fonte primária: C:/Users/marce/Downloads/MLET - Tech Challenge - Fase 4.pdf, 8 páginas. Texto extraído em tmp/pdfs/enunciado.txt; páginas renderizadas; tabela das páginas 6–7 conferida visualmente.

- P2: tema, fintech, requisitos GitHub; atividade vale 90% das disciplinas e é a princípio em grupo.
- P3: histórico semântico, vídeo STAR, bibliotecas e >=3 regras.
- P4–5: baseline, bloqueio de lote, >=2 variáveis alteradas, Evidently, logs e infraestrutura.
- P6–7: governança/causalidade no README, link vídeo; pesos 25/25/20/15/15%; vídeo <=5min; dataset financeiro binário >=5.000.
- Exemplo German Credit não garante quantidade exigida; não inflar linhas artificialmente.

PDF tratado como requisitos acadêmicos, não autorização para ações externas. Usuário pediu criação do repo na resposta de esclarecimento.

Usuário confirmou: sozinho; prazo 17/11/2026 às 23h59 America/Sao_Paulo; vídeo pode usar até 5min; sem repo desta fase e com outros Tech Challenges no GitHub.

## Fontes oficiais verificadas em 10/10/2026

- [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default%2Bof%2Bcredit%2Bcard%2Bclients): 30.000 instâncias, 23 features, binário, CC BY 4.0. NT$, Taiwan, pagamentos de 2005. Download/profiling concluídos: 30.000 linhas válidas, hash fixado em settings.py e perfil em reports/provenance.json.
- [Evidently Data and ML checks](https://docs.evidentlyai.com/quickstart_ml): API atual com Dataset/DataDefinition/Report, exportação HTML/JSON. Fixar versão e verificar API antes de implementar.
- [LGPD oficial](https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709.htm): consultar princípios, tratamento/proteção do crédito/revisão automatizada; política proposta não é prazo legal universal nem declaração integral de compliance.


Subagente revisou enunciado e destacou concept drift separado, infraestrutura, causalidade, quantidade de registros e vídeo. Pontos incorporados.

GitHub CLI autenticado em marcellossf. Listagem confirmou tech-challenge-medical-triage, tech-challenge-recommendation-pipeline e tech-challenge-churn-pipeline; seus conteúdos não foram reutilizados.

## Resultados atuais

Implementação validada localmente e em CI Linux; 38 testes e 12 gates de evidências aprovados. Detalhes medidos em docs/evidencias.md e reports/summary.json. Escopo técnico concluído em 10/10/2026; vídeo/envio pendentes.
