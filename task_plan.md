# Tech Challenge 4 â€” plano completo

Atualizado em 10/10/2026. Entrega individual: **17/11/2026 Ã s 23h59 (America/Sao_Paulo)**. VÃ­deo STAR alvo: **4min30**, limite de 5 minutos. ImplementaÃ§Ã£o iniciada em 10/10/2026 por autorizaÃ§Ã£o do usuÃ¡rio; resultados sÃ³ serÃ£o registrados apÃ³s execuÃ§Ã£o validada.

## Objetivo e stack

Construir sustentaÃ§Ã£o para classificaÃ§Ã£o de crÃ©dito: bloquear dados invÃ¡lidos, simular/detectar data e concept drift, monitorar pipeline/modelo/infraestrutura e documentar governanÃ§a/causalidade.

Proposta: Python 3.11, pandas, scikit-learn, Pandera, Evidently, scipy, MLflow e psutil; pytest e Ruff para checks. Fixar versÃµes apÃ³s instalaÃ§Ã£o e smoke test em ambiente limpo. ExecuÃ§Ã£o local por CLI, com relatÃ³rios HTML/JSON e acompanhamento no MLflow. GitHub Actions para checks e demo reproduzÃ­vel. Cloud, API, Airflow e ONNX nÃ£o sÃ£o necessÃ¡rios para cumprir o escopo bÃ¡sico; Docker pode ser acrescentado depois.

Dataset proposto: UCI Default of Credit Card Clients, 30.000 registros, alvo binÃ¡rio e licenÃ§a CC BY 4.0. Registrar autoria, fonte, licenÃ§a, hash e dicionÃ¡rio. Limite de crÃ©dito nÃ£o Ã© renda; moeda NT$, nÃ£o reais. A base Ã© de Taiwan com histÃ³rico de pagamentos de 2005: nÃ£o representa clientes brasileiros atuais.

## Entregas e avaliaÃ§Ã£o

| CritÃ©rio | Peso | EvidÃªncia de aceite |
|---|---:|---|
| ValidaÃ§Ã£o | 25% | Pandera, pelo menos 3 regras rÃ­gidas; lote vÃ¡lido passa e invÃ¡lido Ã© bloqueado antes da inferÃªncia |
| Drift | 25% | ReferÃªncia separada, duas features alteradas, relatÃ³rio Evidently com estatÃ­stica/limiar e comparaÃ§Ã£o de desempenho |
| Observabilidade | 20% | Logs consolidados; MLflow/HTML com volume, duraÃ§Ã£o, falhas, alertas e saÃºde de infraestrutura |
| GovernanÃ§a | 15% | README com privacidade, base legal contextualizada, retenÃ§Ã£o, vieses e causalidade |
| VÃ­deo STAR | 15% | DemonstraÃ§Ã£o real, link acessÃ­vel e duraÃ§Ã£o <=5min |

TambÃ©m obrigatÃ³rios: GitHub, histÃ³rico semÃ¢ntico/organizado, scripts operantes, classificaÃ§Ã£o financeira binÃ¡ria com >=5.000 amostras. O contexto menciona concept drift: incluir cenÃ¡rio explÃ­cito evita cobrir apenas mudanÃ§a de distribuiÃ§Ã£o.

## Fluxo e separaÃ§Ã£o

Download/proveniÃªncia â†’ preparaÃ§Ã£o/split â†’ contrato â†’ baseline congelado â†’ cenÃ¡rios â†’ validaÃ§Ã£o antes de inferÃªncia â†’ drift/desempenho â†’ logs/MLflow/HTML â†’ resposta documentada.

ReferÃªncia imutÃ¡vel. Cada run registra seed, cenÃ¡rio, versÃ£o/hash de dados e modelo, configuraÃ§Ã£o e run_id. Lote rejeitado termina sem previsÃµes; a demo captura essa falha esperada para seguir com outros cenÃ¡rios. Falhas inesperadas devem terminar com erro.

## ExecuÃ§Ã£o e calendÃ¡rio

### 1. Ambiente e dados â€” 11 a 14/10 â€” 4â€“6h

Criar repo, pyproject.toml, .gitignore e estrutura src/tests/configs/docs. Download com cache e hash, sem dados brutos no Git. Inspecionar tipos, domÃ­nios, nulos, duplicatas e contagens; registrar normalizaÃ§Ã£o/linhas rejeitadas. NÃ£o corrigir cÃ³digos por suposiÃ§Ã£o.

Proposta de split estratificado por ID e seed fixa: 60% referÃªncia/treino, 20% validaÃ§Ã£o e 20% produÃ§Ã£o simulada (18.000/6.000/6.000 antes da limpeza). Zero sobreposiÃ§Ã£o. Ajustar contagens apÃ³s limpeza e manter >=5.000 registros reais. Preprocessamento ajustado somente no treino. Documentar que perÃ­odos sÃ£o simulados, nÃ£o fluxo temporal real.

Aceite: fonte/licenÃ§a/hash e contagens registradas, alvo binÃ¡rio, rotina reproduzÃ­vel e teste de ausÃªncia de sobreposiÃ§Ã£o. NÃ£o duplicar linhas para satisfazer quantidade.

### 2. Baseline â€” 15 a 17/10 â€” 3â€“5h

Pipeline scikit-learn e LogisticRegression; Ã¡rvore simples como alternativa justificada. Excluir ID/alvo dos preditores. Comparar com modelo constante. Medir ROC-AUC, PR-AUC, precision, recall, F1, matriz de confusÃ£o e Brier score. Escolher limiar usando validaÃ§Ã£o e congelar antes dos cenÃ¡rios.

Salvar modelo, features, seed, parÃ¢metros, hashes e mÃ©tricas. Foco Ã© sustentaÃ§Ã£o, nÃ£o maximizaÃ§Ã£o de acurÃ¡cia.

Aceite: modelo carrega e gera probabilidades vÃ¡lidas; mÃ©tricas reais e baseline ingÃªnuo documentados; sem vazamento de produÃ§Ã£o.

### 3. Contratos e bloqueio â€” 18 a 21/10 â€” 4â€“6h

Pandera: colunas/tipos obrigatÃ³rios, ausÃªncia de nulos, idade em faixa justificada, limite de crÃ©dito positivo, chave Ãºnica. Conferir domÃ­nios com profiling. Faturas negativas e cÃ³digos negativos de pagamento podem ser legÃ­timos: nÃ£o proibir sem verificar o dicionÃ¡rio.

Separar contrato de inferÃªncia sem alvo obrigatÃ³rio de contrato de avaliaÃ§Ã£o com alvo 0/1. Criar lote com nulo, tipo incorreto, idade impossÃ­vel e ID duplicado. Bloquear lote inteiro antes de prever; registrar erros sanitizados, sem valores individuais.

Aceite: lote vÃ¡lido passa, invÃ¡lido falha, inferÃªncia nÃ£o Ã© chamada apÃ³s rejeiÃ§Ã£o; quality_errors.json mostra regra/contagem. Testar esses comportamentos.

### 4. SimulaÃ§Ãµes â€” 22 a 26/10 â€” 5â€“7h

Gerar cenÃ¡rios com seed e manifesto de transformaÃ§Ã£o:

1. Controle saudÃ¡vel: produÃ§Ã£o reservada sem intervenÃ§Ã£o; medir variaÃ§Ã£o natural/falsos alertas.
2. InvÃ¡lido: falha contratual prevista, nÃ£o usado como demonstraÃ§Ã£o de drift.
3. Data drift: alterar gradualmente LIMIT_BAL e PAY_AMT1 em janelas t1/t2/t3, mantendo domÃ­nios vÃ¡lidos. Confirmar relevÃ¢ncia no baseline; se necessÃ¡rio, escolher outras duas features antes de congelar o desenho. Declarar fatores e preservar referÃªncia. NÃ£o chamar cenÃ¡rio artificial de economia real.
4. Concept drift: partir da produÃ§Ã£o limpa, preservar X e alterar probabilisticamente y em segmento declarado, com intensidade e seed configuradas. Rotular resultados como sintÃ©ticos; comparar desempenho do mesmo modelo congelado contra esses labels.

Aceite: duas features com mudanÃ§a verificÃ¡vel, transformaÃ§Ãµes reproduzÃ­veis, cenÃ¡rios separados, concept drift com mecanismo e rÃ³tulos conhecidos. Alterar X ou observar target drift nÃ£o prova concept drift. Se o efeito planejado nÃ£o aparecer, investigar e documentar revisÃ£o; nÃ£o editar mÃ©tricas manualmente.

### 5. DetecÃ§Ã£o e relatÃ³rios â€” 27 a 30/10 â€” 4â€“6h

Integrar Evidently na API da versÃ£o fixada, exportar HTML/JSON. PSI com bins da referÃªncia e tratamento explÃ­cito de zeros; KS para variÃ¡veis numÃ©ricas; mÃ©todo adequado para categorias. Mostrar amostra, efeito, estatÃ­stica, p-valor quando aplicÃ¡vel e limiar.

Propostas iniciais, a calibrar no controle e justificar: PSI >=0,20 como alerta, KS alpha=0,01 com correÃ§Ã£o de testes mÃºltiplos quando aplicÃ¡vel; queda de recall >=0,10 absoluta como candidato a alerta de desempenho. NÃ£o sÃ£o requisitos do PDF nem padrÃµes universais. Acompanhar tambÃ©m PR-AUC e Brier. Sem labels, desempenho deve aparecer como indisponÃ­vel, nÃ£o saudÃ¡vel.

Aceite: HTML abre e identifica features manipuladas, controle disponÃ­vel, queda de desempenho medida com labels; limiares versionados antes da execuÃ§Ã£o final.

### 6. Observabilidade â€” 31/10 a 04/11 â€” 5â€“7h

Logs JSONL com run_id, cenÃ¡rio, estÃ¡gio, status, timestamps, duraÃ§Ã£o, volume, rejeiÃ§Ãµes, drift, alerta e versÃµes. NÃ£o logar linhas individuais. MLflow registra parÃ¢metros/mÃ©tricas/artefatos dos cenÃ¡rios e rejeiÃ§Ã£o esperada.

Coletar CPU, memÃ³ria e espaÃ§o livre com psutil; amostrar durante execuÃ§Ã£o e documentar intervalo/limitaÃ§Ãµes. HTML consolidado mostra pipeline, infraestrutura, modelo e alertas por janela. MLflow sozinho com apenas mÃ©tricas de treino nÃ£o cobre saÃºde completa.

Aceite: sucesso/falha visÃ­veis, alertas rastreÃ¡veis, infraestrutura realmente medida, um comando reproduz evidÃªncias.

### 7. GovernanÃ§a, causalidade e integraÃ§Ã£o â€” 05 a 08/11 â€” 4â€“6h

README consolidado: problema, arquitetura, instalaÃ§Ã£o, comandos testados, fonte/licenÃ§a, contrato, model card, simulaÃ§Ãµes, mÃ©tricas, limitaÃ§Ãµes, governanÃ§a e links. Documentos auxiliares podem estar em docs, mas o plano de governanÃ§a precisa constar no README.

Minimizar dados: retirar ID de origem do modelo, chave tÃ©cnica local e relatÃ³rios agregados. Remover ID nÃ£o garante anonimizaÃ§Ã£o. Justificar uso/exclusÃ£o de sexo, estado civil e escolaridade; avaliar diferenÃ§as de erro por subgrupos com amostra suficiente, sem concluir equidade universal.

Separar uso acadÃªmico de fintech real. Contextualizar princÃ­pios (art. 6Âº), proteÃ§Ã£o do crÃ©dito (art. 7Âº, X) e revisÃ£o de decisÃµes automatizadas (art. 20). Base legal do tratamento acadÃªmico exige anÃ¡lise prÃ³pria; licenÃ§a pÃºblica nÃ£o a substitui. NÃ£o afirmar certificaÃ§Ã£o/compliance integral.

RetenÃ§Ã£o proposta como polÃ­tica acadÃªmica, nÃ£o prazo legal: cache atÃ© avaliaÃ§Ã£o, excluir atÃ© 30 dias apÃ³s encerramento definido pela instituiÃ§Ã£o; relatÃ³rios agregados por 90 dias se necessÃ¡rios. Documentar responsÃ¡vel, acesso, backups, descarte e canal de revisÃ£o hipotÃ©tico.

Causalidade: hipÃ³tese choque econÃ´mico â†’ capacidade de pagamento â†’ inadimplÃªncia; intervenÃ§Ãµes simuladas conhecidas, possÃ­veis confundidores e limites de extrapolaÃ§Ã£o. KS/PSI nÃ£o provam causa econÃ´mica em produÃ§Ã£o. Resposta: bloquear invÃ¡lidos, investigar drift, obter labels, revisar segmento e validar retraining antes de promover; sem reentreino automÃ¡tico por qualquer alerta.

Aceite: README cobre tÃ³picos, cada resultado possui artefato; testes relevantes de contrato/split/cenÃ¡rios/integraÃ§Ã£o e CI passam em ambiente limpo; acesso do avaliador funciona.

### 8. EvidÃªncias e vÃ­deo â€” 09 a 13/11 â€” 3â€“4h

Demo do zero na versÃ£o final; conferir HTML/MLflow; guardar logs/screenshot agregados e run_id. Ensaiar 4min30 duas vezes; gravar, assistir inteiro e verificar Ã¡udio/legibilidade. Publicar link acessÃ­vel e testar sem sessÃ£o autenticada. NÃ£o mostrar tokens ou dados individuais.

Aceite: <=5min, quatro blocos STAR, resultados correspondentes Ã  versÃ£o do repo.

### 9. Margem e entrega â€” 14 a 17/11 â€” 2â€“3h

14â€“16/11: correÃ§Ãµes e revisÃ£o de links/reproduÃ§Ã£o/gravaÃ§Ã£o. Meta de envio: **16/11 atÃ© 20h**. Em 17/11 conferir recibo e acesso antes de 23h59. Consultar no portal o formato exato de submissÃ£o.

Total estimado: **34â€“50 horas**, incluindo margem. Com menos disponibilidade, simplificar aparÃªncia/empacotamento preservando contratos, drift e observabilidade.

## Estrutura proposta, ainda nÃ£o implementada

```text
README.md
pyproject.toml
.github/workflows/ci.yml
configs/demo.yaml
src/credit_monitoring/{data,contracts,train,simulate,drift,monitoring,pipeline,cli}.py
tests/
docs/{data_dictionary,model_card,causalidade,evidencias}.md
reports/          # HTML/JSON agregados publicÃ¡veis
data/             # ignorado; download reproduzÃ­vel
artifacts/        # ignorado ou release
```

Interface proposta, ainda indisponÃ­vel: `python -m credit_monitoring.cli download`, `train`, `demo`, `verify`. Publicar instruÃ§Ãµes executÃ¡veis somente depois de implementadas/testadas.

## Commits reais por etapa

Janelas sÃ£o propostas de sessÃµes de trabalho, nÃ£o datas para adulterar histÃ³rico. Commit ao terminar/testar/revisar entrega; sem espera artificial, datas forjadas ou encenaÃ§Ã£o de autoria. Adaptar mensagens ao que mudou. Declarar assistÃªncia conforme regras da instituiÃ§Ã£o e explicar com suas palavras.

| Janela | Mensagem sugerida | VerificaÃ§Ã£o |
|---|---|---|
| 10â€“14/10 | docs: planejar entrega da fase 4 | Plano e cronograma |
| 11â€“14/10 | chore: configurar ambiente e estrutura | InstalaÃ§Ã£o limpa |
| 11â€“14/10 | feat: preparar base e separar dados | Hash, contagens, zero sobreposiÃ§Ã£o |
| 15â€“17/10 | feat: treinar baseline de inadimplÃªncia | Artefato e mÃ©tricas |
| 18â€“21/10 | feat: bloquear lotes invÃ¡lidos com Pandera | Teste vÃ¡lido/invÃ¡lido |
| 22â€“26/10 | feat: simular data drift e concept drift | Manifestos/seed |
| 27â€“30/10 | feat: gerar relatÃ³rios de drift e desempenho | HTML e estatÃ­sticas |
| 31/10â€“04/11 | feat: consolidar logs mÃ©tricas e alertas | MLflow e infraestrutura |
| 05â€“08/11 | docs: documentar governanÃ§a e causalidade | README completo |
| 05â€“08/11 | ci: validar pipeline e evidÃªncias | CI e reproduÃ§Ã£o |
| 09â€“13/11 | docs: adicionar evidÃªncias e vÃ­deo STAR | Links e duraÃ§Ã£o |

CorreÃ§Ãµes reais geram `fix:`; nÃ£o fabricar bugs para gerar histÃ³rico. Antes de cada etapa, responder: o que mudou, por que essa escolha e como a evidÃªncia comprova funcionamento?

## Status e pendÃªncias

- [x] PDF lido e rubrica conferida visualmente.
- [x] Prazo/individual/vÃ­deo confirmados.
- [x] RevisÃ£o independente com subagente.
- [x] RepositÃ³rio pÃºblico, conforme autorizaÃ§Ã£o de 10/10/2026: https://github.com/marcellossf/tech-challenge-credit-monitoring
- [x] ImplementaÃ§Ã£o tÃ©cnica das etapas 1â€“7 executada e validada localmente em 10/10/2026.
- [ ] CI remoto, evidÃªncias finais e revisÃ£o de apresentaÃ§Ã£o.
- [ ] VÃ­deo e submissÃ£o institucional.

PendÃªncias: disponibilidade semanal; confirmaÃ§Ã£o institucional para entrega individual (PDF diz a princÃ­pio em grupo); regras de assistÃªncia IA; portal/formato de envio. NÃ£o impedem planejamento.

OcorrÃªncias: runtime sem fitz, resolvido com pypdf/pypdfium2; impressÃ£o cp1252 falhou, arquivo UTF-8 lido; Git inicialmente sem commits/remote; patch inicial rejeitado por operaÃ§Ãµes duplicadas no mesmo arquivo, refeito sem alteraÃ§Ã£o parcial.

## ExecuÃ§Ã£o iniciada em 10/10/2026

UsuÃ¡rio autorizou tornar o repo pÃºblico e comeÃ§ar. Cronograma Ã© referÃªncia; etapas adiantadas serÃ£o registradas com datas reais. DisponÃ­vel Python 3.13.6; ambiente isolado serÃ¡ validado nessa versÃ£o, suporte declarado >=3.11,<3.14. TrÃªs subagentes implementam contratos, drift e observabilidade em arquivos separados; root integra dados/baseline/pipeline/CLI.
