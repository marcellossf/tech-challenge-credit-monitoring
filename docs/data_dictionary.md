# Dados e contrato

Fonte: [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default%2Bof%2Bcredit%2Bcard%2Bclients), I-Cheng Yeh, CC BY 4.0, DOI 10.24432/C55S3H. Pagamentos de Taiwan/2005 e inadimplência no mês seguinte. Cinco preditores selecionados das 23 variáveis.

| Campo | Origem | Significado | Contrato |
|---|---|---|---|
| row_id | ID | Chave só local | Inteiro positivo único, nunca feature |
| AGE | X5 | Idade anos | Inteiro 18–100 |
| LIMIT_BAL | X1 | Limite NT$, não renda | >0 e <=2.000.000 |
| PAY_0 | X6 | Status pagamento setembro | Inteiro -2..9 |
| BILL_AMT1 | X12 | Fatura setembro NT$ | Finito, pode ser negativo |
| PAY_AMT1 | X18 | Pagamento setembro NT$ | Finito não negativo |
| target | default payment next month | Inadimplência | Inteiro 0/1 |

Faixas são políticas, não aprendidas do treino. Fonte inesperada rejeitada; sem descarte silencioso. PAY_0 -2/-1/0 são códigos, não atraso negativo; uso contínuo na regressão é simplificação. Perfil real em reports/provenance.json.

Sexo/escolaridade/estado civil excluídos. Remover ID não garante anonimização. Split seed42 60/20/20 sem sobreposição, não valida generalização temporal. Janelas usam mesmo holdout para isolar intervenções.
