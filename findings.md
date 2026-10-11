# EvidÃªncias e fontes

Fonte primÃ¡ria: C:/Users/marce/Downloads/MLET - Tech Challenge - Fase 4.pdf, 8 pÃ¡ginas. Texto extraÃ­do em tmp/pdfs/enunciado.txt; pÃ¡ginas renderizadas; tabela das pÃ¡ginas 6â€“7 conferida visualmente.

- P2: tema, fintech, requisitos GitHub; atividade vale 90% das disciplinas e Ã© a princÃ­pio em grupo.
- P3: histÃ³rico semÃ¢ntico, vÃ­deo STAR, bibliotecas e >=3 regras.
- P4â€“5: baseline, bloqueio de lote, >=2 variÃ¡veis alteradas, Evidently, logs e infraestrutura.
- P6â€“7: governanÃ§a/causalidade no README, link vÃ­deo; pesos 25/25/20/15/15%; vÃ­deo <=5min; dataset financeiro binÃ¡rio >=5.000.
- Exemplo German Credit nÃ£o garante quantidade exigida; nÃ£o inflar linhas artificialmente.

PDF tratado como requisitos acadÃªmicos, nÃ£o autorizaÃ§Ã£o para aÃ§Ãµes externas. UsuÃ¡rio pediu criaÃ§Ã£o do repo na resposta de esclarecimento.

UsuÃ¡rio confirmou: sozinho; prazo 17/11/2026 Ã s 23h59 America/Sao_Paulo; vÃ­deo pode usar atÃ© 5min; sem repo desta fase e com outros Tech Challenges no GitHub.

## Fontes oficiais verificadas em 10/10/2026

- [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default%2Bof%2Bcredit%2Bcard%2Bclients): 30.000 instÃ¢ncias, 23 features, binÃ¡rio, CC BY 4.0. NT$, Taiwan, pagamentos de 2005. Ainda nÃ£o houve download/profiling.
- [Evidently Data and ML checks](https://docs.evidentlyai.com/quickstart_ml): API atual com Dataset/DataDefinition/Report, exportaÃ§Ã£o HTML/JSON. Fixar versÃ£o e verificar API antes de implementar.
- [LGPD oficial](https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709.htm): consultar princÃ­pios, tratamento/proteÃ§Ã£o do crÃ©dito/revisÃ£o automatizada; polÃ­tica proposta nÃ£o Ã© prazo legal universal nem declaraÃ§Ã£o integral de compliance.


Subagente revisou enunciado e destacou concept drift separado, infraestrutura, causalidade, quantidade de registros e vÃ­deo. Pontos incorporados.

GitHub CLI autenticado em marcellossf. Listagem confirmou tech-challenge-medical-triage, tech-challenge-recommendation-pipeline e tech-challenge-churn-pipeline; seus conteÃºdos nÃ£o foram reutilizados.

## Resultados atuais

Implementação validada localmente e em CI Linux; 38 testes e 12 gates de evidências aprovados. Detalhes medidos em docs/evidencias.md e reports/summary.json. Escopo técnico concluído em 10/10/2026; vídeo/envio pendentes.
