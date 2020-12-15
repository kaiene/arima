Essas rotinas foram desenvolvidas para um trabalho da disciplina de ITS, Poli-USP 2020.

Trata-se da Aplicação do ARIMA em Previsão de Tráfego de Curto Prazo.

A base de dados consistia em dados de volume de tráfego obtidos de radares, agrupados em 1, 2, 3, 4 e 5 minutos.

Há dois modelos:

MODELO 1: Modelo com dados de treino consecutivos, isto é, utiliza como base todos 
os dias anteriores da semana para a previsão

MODELO 2: Utiliza como base somente os dias da semana do dia escolhido para a previsão.
Por exemplo: O dia escolhido cai em uma quarta-feira, então baseia-se em todas as quartas-feiras
do banco de dados para realizar a previsão.

Os arquivos de dados consistiam em volumes de tráfego agrupados, onde a coluna com as quantidades continha a soma dos dados de cada instante do agrupamento.

O arquivo calendario.png utilizado refere-se ao mês de março de 2018, e só é necessário para facilitar a escolha dos inputs de dia da semana
pelo usuário.

Qualquer dúvida, contato: kaiene.paz@usp.br e michelle.trintinalia@usp.br