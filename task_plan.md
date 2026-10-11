# Tech Challenge 4 — plano completo

Atualizado em 10/10/2026. Entrega individual: **17/11/2026 às 23h59 (America/Sao_Paulo)**. Vídeo STAR alvo: **4min30**, limite de 5 minutos. Implementação iniciada em 10/10/2026 por autorização do usuário; resultados só serão registrados após execução validada.

## Objetivo e stack

Construir sustentação para classificação de crédito: bloquear dados inválidos, simular/detectar data e concept drift, monitorar pipeline/modelo/infraestrutura e documentar governança/causalidade.

Proposta: Python 3.11, pandas, scikit-learn, Pandera, Evidently, scipy, MLflow e psutil; pytest e Ruff para checks. Fixar versões após instalação e smoke test em ambiente limpo. Execução local por CLI, com relatórios HTML/JSON e acompanhamento no MLflow. GitHub Actions para checks e demo reproduzível. Cloud, API, Airflow e ONNX não são necessários para cumprir o escopo básico; Docker pode ser acrescentado depois.

Dataset proposto: UCI Default of Credit Card Clients, 30.000 registros, alvo binário e licença CC BY 4.0. Registrar autoria, fonte, licença, hash e dicionário. Limite de crédito não é renda; moeda NT$, não reais. A base é de Taiwan com histórico de pagamentos de 2005: não representa clientes brasileiros atuais.

## Entregas e avaliação

| Critério | Peso | Evidência de aceite |
|---|---:|---|
| Validação | 25% | Pandera, pelo menos 3 regras rígidas; lote válido passa e inválido é bloqueado antes da inferência |
| Drift | 25% | Referência separada, duas features alteradas, relatório Evidently com estatística/limiar e comparação de desempenho |
| Observabilidade | 20% | Logs consolidados; MLflow/HTML com volume, duração, falhas, alertas e saúde de infraestrutura |
| Governança | 15% | README com privacidade, base legal contextualizada, retenção, vieses e causalidade |
| Vídeo STAR | 15% | Demonstração real, link acessível e duração <=5min |

Também obrigatórios: GitHub, histórico semântico/organizado, scripts operantes, classificação financeira binária com >=5.000 amostras. O contexto menciona concept drift: incluir cenário explícito evita cobrir apenas mudança de distribuição.

## Fluxo e separação

Download/proveniência → preparação/split → contrato → baseline congelado → cenários → validação antes de inferência → drift/desempenho → logs/MLflow/HTML → resposta documentada.

Referência imutável. Cada run registra seed, cenário, versão/hash de dados e modelo, configuração e run_id. Lote rejeitado termina sem previsões; a demo captura essa falha esperada para seguir com outros cenários. Falhas inesperadas devem terminar com erro.

## Execução e calendário

### 1. Ambiente e dados — 11 a 14/10 — 4–6h

Criar repo, pyproject.toml, .gitignore e estrutura src/tests/configs/docs. Download com cache e hash, sem dados brutos no Git. Inspecionar tipos, domínios, nulos, duplicatas e contagens; registrar normalização/linhas rejeitadas. Não corrigir códigos por suposição.

Proposta de split estratificado por ID e seed fixa: 60% referência/treino, 20% validação e 20% produção simulada (18.000/6.000/6.000 antes da limpeza). Zero sobreposição. Ajustar contagens após limpeza e manter >=5.000 registros reais. Preprocessamento ajustado somente no treino. Documentar que períodos são simulados, não fluxo temporal real.

Aceite: fonte/licença/hash e contagens registradas, alvo binário, rotina reproduzível e teste de ausência de sobreposição. Não duplicar linhas para satisfazer quantidade.

### 2. Baseline — 15 a 17/10 — 3–5h

Pipeline scikit-learn e LogisticRegression; árvore simples como alternativa justificada. Excluir ID/alvo dos preditores. Comparar com modelo constante. Medir ROC-AUC, PR-AUC, precision, recall, F1, matriz de confusão e Brier score. Escolher limiar usando validação e congelar antes dos cenários.

Salvar modelo, features, seed, parâmetros, hashes e métricas. Foco é sustentação, não maximização de acurácia.

Aceite: modelo carrega e gera probabilidades válidas; métricas reais e baseline ingênuo documentados; sem vazamento de produção.

### 3. Contratos e bloqueio — 18 a 21/10 — 4–6h

Pandera: colunas/tipos obrigatórios, ausência de nulos, idade em faixa justificada, limite de crédito positivo, chave única. Conferir domínios com profiling. Faturas negativas e códigos negativos de pagamento podem ser legítimos: não proibir sem verificar o dicionário.

Separar contrato de inferência sem alvo obrigatório de contrato de avaliação com alvo 0/1. Criar lote com nulo, tipo incorreto, idade impossível e ID duplicado. Bloquear lote inteiro antes de prever; registrar erros sanitizados, sem valores individuais.

Aceite: lote válido passa, inválido falha, inferência não é chamada após rejeição; quality_errors.json mostra regra/contagem. Testar esses comportamentos.

### 4. Simulações — 22 a 26/10 — 5–7h

Gerar cenários com seed e manifesto de transformação:

1. Controle saudável: produção reservada sem intervenção; medir variação natural/falsos alertas.
2. Inválido: falha contratual prevista, não usado como demonstração de drift.
3. Data drift: alterar gradualmente LIMIT_BAL e PAY_AMT1 em janelas t1/t2/t3, mantendo domínios válidos. Confirmar relevância no baseline; se necessário, escolher outras duas features antes de congelar o desenho. Declarar fatores e preservar referência. Não chamar cenário artificial de economia real.
4. Concept drift: partir da produção limpa, preservar X e alterar probabilisticamente y em segmento declarado, com intensidade e seed configuradas. Rotular resultados como sintéticos; comparar desempenho do mesmo modelo congelado contra esses labels.

Aceite: duas features com mudança verificável, transformações reproduzíveis, cenários separados, concept drift com mecanismo e rótulos conhecidos. Alterar X ou observar target drift não prova concept drift. Se o efeito planejado não aparecer, investigar e documentar revisão; não editar métricas manualmente.

### 5. Detecção e relatórios — 27 a 30/10 — 4–6h

Integrar Evidently na API da versão fixada, exportar HTML/JSON. PSI com bins da referência e tratamento explícito de zeros; KS para variáveis numéricas; método adequado para categorias. Mostrar amostra, efeito, estatística, p-valor quando aplicável e limiar.

Propostas iniciais, a calibrar no controle e justificar: PSI >=0,20 como alerta, KS alpha=0,01 com correção de testes múltiplos quando aplicável; queda de recall >=0,10 absoluta como candidato a alerta de desempenho. Não são requisitos do PDF nem padrões universais. Acompanhar também PR-AUC e Brier. Sem labels, desempenho deve aparecer como indisponível, não saudável.

Aceite: HTML abre e identifica features manipuladas, controle disponível, queda de desempenho medida com labels; limiares versionados antes da execução final.

### 6. Observabilidade — 31/10 a 04/11 — 5–7h

Logs JSONL com run_id, cenário, estágio, status, timestamps, duração, volume, rejeições, drift, alerta e versões. Não logar linhas individuais. MLflow registra parâmetros/métricas/artefatos dos cenários e rejeição esperada.

Coletar CPU, memória e espaço livre com psutil; amostrar durante execução e documentar intervalo/limitações. HTML consolidado mostra pipeline, infraestrutura, modelo e alertas por janela. MLflow sozinho com apenas métricas de treino não cobre saúde completa.

Aceite: sucesso/falha visíveis, alertas rastreáveis, infraestrutura realmente medida, um comando reproduz evidências.

### 7. Governança, causalidade e integração — 05 a 08/11 — 4–6h

README consolidado: problema, arquitetura, instalação, comandos testados, fonte/licença, contrato, model card, simulações, métricas, limitações, governança e links. Documentos auxiliares podem estar em docs, mas o plano de governança precisa constar no README.

Minimizar dados: retirar ID de origem do modelo, chave técnica local e relatórios agregados. Remover ID não garante anonimização. Justificar uso/exclusão de sexo, estado civil e escolaridade; avaliar diferenças de erro por subgrupos com amostra suficiente, sem concluir equidade universal.

Separar uso acadêmico de fintech real. Contextualizar princípios (art. 6º), proteção do crédito (art. 7º, X) e revisão de decisões automatizadas (art. 20). Base legal do tratamento acadêmico exige análise própria; licença pública não a substitui. Não afirmar certificação/compliance integral.

Retenção proposta como política acadêmica, não prazo legal: cache até avaliação, excluir até 30 dias após encerramento definido pela instituição; relatórios agregados por 90 dias se necessários. Documentar responsável, acesso, backups, descarte e canal de revisão hipotético.

Causalidade: hipótese choque econômico → capacidade de pagamento → inadimplência; intervenções simuladas conhecidas, possíveis confundidores e limites de extrapolação. KS/PSI não provam causa econômica em produção. Resposta: bloquear inválidos, investigar drift, obter labels, revisar segmento e validar retraining antes de promover; sem reentreino automático por qualquer alerta.

Aceite: README cobre tópicos, cada resultado possui artefato; testes relevantes de contrato/split/cenários/integração e CI passam em ambiente limpo; acesso do avaliador funciona.

### 8. Evidências e vídeo — 09 a 13/11 — 3–4h

Demo do zero na versão final; conferir HTML/MLflow; guardar logs/screenshot agregados e run_id. Ensaiar 4min30 duas vezes; gravar, assistir inteiro e verificar áudio/legibilidade. Publicar link acessível e testar sem sessão autenticada. Não mostrar tokens ou dados individuais.

Aceite: <=5min, quatro blocos STAR, resultados correspondentes à versão do repo.

### 9. Margem e entrega — 14 a 17/11 — 2–3h

14–16/11: correções e revisão de links/reprodução/gravação. Meta de envio: **16/11 até 20h**. Em 17/11 conferir recibo e acesso antes de 23h59. Consultar no portal o formato exato de submissão.

Total estimado: **34–50 horas**, incluindo margem. Com menos disponibilidade, simplificar aparência/empacotamento preservando contratos, drift e observabilidade.

## Estrutura proposta, ainda não implementada

```text
README.md
pyproject.toml
.github/workflows/ci.yml
configs/demo.yaml
src/credit_monitoring/{data,contracts,train,simulate,drift,monitoring,pipeline,cli}.py
tests/
docs/{data_dictionary,model_card,causalidade,evidencias}.md
reports/          # HTML/JSON agregados publicáveis
data/             # ignorado; download reproduzível
artifacts/        # ignorado ou release
```

Interface proposta, ainda indisponível: `python -m credit_monitoring.cli download`, `train`, `demo`, `verify`. Publicar instruções executáveis somente depois de implementadas/testadas.

## Commits reais por etapa

Janelas são propostas de sessões de trabalho, não datas para adulterar histórico. Commit ao terminar/testar/revisar entrega; sem espera artificial, datas forjadas ou encenação de autoria. Adaptar mensagens ao que mudou. Declarar assistência conforme regras da instituição e explicar com suas palavras.

| Janela | Mensagem sugerida | Verificação |
|---|---|---|
| 10–14/10 | docs: planejar entrega da fase 4 | Plano e cronograma |
| 11–14/10 | chore: configurar ambiente e estrutura | Instalação limpa |
| 11–14/10 | feat: preparar base e separar dados | Hash, contagens, zero sobreposição |
| 15–17/10 | feat: treinar baseline de inadimplência | Artefato e métricas |
| 18–21/10 | feat: bloquear lotes inválidos com Pandera | Teste válido/inválido |
| 22–26/10 | feat: simular data drift e concept drift | Manifestos/seed |
| 27–30/10 | feat: gerar relatórios de drift e desempenho | HTML e estatísticas |
| 31/10–04/11 | feat: consolidar logs métricas e alertas | MLflow e infraestrutura |
| 05–08/11 | docs: documentar governança e causalidade | README completo |
| 05–08/11 | ci: validar pipeline e evidências | CI e reprodução |
| 09–13/11 | docs: adicionar evidências e vídeo STAR | Links e duração |

Correções reais geram `fix:`; não fabricar bugs para gerar histórico. Antes de cada etapa, responder: o que mudou, por que essa escolha e como a evidência comprova funcionamento?

## Status e pendências

- [x] PDF lido e rubrica conferida visualmente.
- [x] Prazo/individual/vídeo confirmados.
- [x] Revisão independente com subagente.
- [x] Repositório público, conforme autorização de 10/10/2026: https://github.com/marcellossf/tech-challenge-credit-monitoring
- [x] Implementação técnica das etapas 1–7 executada e validada localmente em 10/10/2026.
- [ ] Vídeo e submissão institucional.

Pendências: disponibilidade semanal; confirmação institucional para entrega individual (PDF diz a princípio em grupo); regras de assistência IA; portal/formato de envio. Não impedem planejamento.

Ocorrências: runtime sem fitz, resolvido com pypdf/pypdfium2; impressão cp1252 falhou, arquivo UTF-8 lido; Git inicialmente sem commits/remote; patch inicial rejeitado por operações duplicadas no mesmo arquivo, refeito sem alteração parcial.

## Execução iniciada em 10/10/2026

Usuário autorizou tornar o repo público e começar. Cronograma é referência; etapas adiantadas serão registradas com datas reais. Disponível Python 3.13.6; ambiente isolado será validado nessa versão, suporte declarado >=3.11,<3.14. Três subagentes implementam contratos, drift e observabilidade em arquivos separados; root integra dados/baseline/pipeline/CLI.
