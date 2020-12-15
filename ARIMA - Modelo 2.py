# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 17:10:28 2020

@author: Michelle e Kaiene

MODELO 2 = Utiliza como base somente os dias da semana do dia escolhido para a previsão.
Por exemplo: O dia escolhido cai em uma quarta-feira, então baseia-se em todas as quartas-feiras
do banco de dados para realizar a previsão
"""

import pandas as pd #Pacote para lidar com data frames
import numpy as np
import os
from IPython.display import display, Image #Pacote para lidar com figuras
import matplotlib.pyplot as plt #Pacote para lidar com gráficos
from datetime import timedelta, date
import datetime
import statistics
from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.dates as mdates
#from pySTARMA import starma_model as sm

def listagem_valores(coluna):
    #Função recebe uma coluna como array e diz quais os diferentes tipos de valores diferentes existem dentro dela
    lista = []
    tamanho = len(coluna)
    for i in range(1,tamanho):
        if coluna[i] not in lista:
            lista.append(coluna[i])
    return lista

def corta_tempo(dataframe,t1):

    division=dataframe.loc[dataframe['Data']==t1].index[0]

    return dataframe.loc[division:]

def padroniza_dia(dia):
    #Função recebe um valor do dia do mês e transforma em um string data de dois dígitos
    if dia in range(0,10):
        data = str("0"+str(dia))
    else:
        data = str(dia)
        
    return data

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

def get_day(d):
    d = str(d)[8:] #seleciona apenas os dias
    if d[0] == "0":
        d = d[1]
    return d

def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    #acf1 = acf(fc-test)[1]                      # ACF1
    print("MAPE = %f \n ME = %f \n MAE = %f \n MPE = %f \n RMSE = %f \n CORR = %f \n MIN-MAX = %f" %(mape, me, mae, mpe, rmse, corr, minmax))
    return()
            
def main():
    #Recebe o dia da semana escolhido
    display(Image(filename='calendario.png')) #Colocar a imagem do calendário na mesma pasta do arquivo python
    diaok = False
    while not diaok:
        print("0 - Segunda-Feira \n 1 - Terça-Feira \n 2 - Quarta-Feira \n 3 - Quinta-Feira \n 4 - Sexta-Feira \n 5 - Sábado \n 6 - Domingo")
        dia = int(input("Digite o índice do dia da semana desejado: "))
        if dia == 1 or dia == 2 or dia == 3 or dia == 4 or dia == 5 or dia ==6 or dia == 7:
            diaok = True
            print("================================================")
        else:
            print("Input inválido, escolha outro valor")
            print("================================================")
    
    #Cria lista com todos os dias do mês que forem iguais ao dia da semana escolhido
    start_date = date(2018, 3, 1)
    end_date = date(2018, 3, 31)
    lista_dias = []
    for d in daterange(start_date, end_date):
        if d.weekday() == dia:
            lista_dias.append(d)
    
    #Recebe o dia do mês a ser previsto
    for i, d in enumerate(lista_dias):
        print('\n{} - {}'.format(i, d))
        #Printa a lista dos dias disponíveis
    diaok = False
    while not diaok:
        posicao_dia = int(input("Escolha qual dos dias você prefere dentro da lista acima (apenas o índice): "))
        if posicao_dia in range(0, len(lista_dias)):
            diaok = True
            print("================================================")
        else:
            print("Input inválido, escolha outro valor de dia")
            print("================================================")
    
    #Recebe o agrupamento temporal escolhido
    agrupamentook = False
    while not agrupamentook:
        print("Agrupamentos disponíveis: \n1Min \n2Min \n3Min \n4Min \n5Min \n")
        agrupamento = input("Digite o agrupamento temporal desejado: ")
        if agrupamento == "1Min" or agrupamento == "2Min" or agrupamento == "3Min" or agrupamento == "4Min" or agrupamento == "5Min":
            agrupamentook = True
            print("================================================")
        else:
            print("Input inválido, escolha outro valor de agrupamento")
            print("================================================")
    
    #Leitura do arquivo - Alterar com base na localização dos dados do seu computador
    path = "C:\\Users\\walmart\\Documents\\USP\\TCC\\Dados\\Dados Agrupados\\Grouped by frequency"

    nova_lista_dias = [] #Nova lista reordenando os dias para que o escolhido seja o último
    for i in range (0,len(lista_dias)):
        if i != posicao_dia:
            nova_lista_dias.append(lista_dias[i])
    nova_lista_dias.append(lista_dias[posicao_dia])
    
    #Leitura e concatenação dos arquivos dos dias
    for d in nova_lista_dias:
        dia = get_day(d)
        if d == nova_lista_dias[0]:
            dados = pd.read_csv(path + "\\"+ dia +"\\" + dia + "_group_" + str(agrupamento) + ".csv")
        else:
            df = pd.read_csv(path + "\\"+ dia +"\\" + dia + "_group_" + str(agrupamento) + ".csv")
            dados = pd.concat([dados,df])  
    
    dados.Data = pd.to_datetime(dados.Data) #Transforma a coluna Data em formato Data
    
    #Recebe o radar escolhido para previsão    
    radares = [10426, 10433, 10482, 10484, 10492, 10500, 10521, 10531]
    radarok = False
    while not radarok:
        print("Radares disponíveis:", radares, sep='\n' )
        radar = int(input("Qual o radar a ser previsto?: "))
        if radar in radares:
            radarok = True
            print("================================================")
        else:
            print("Input inválido, escolha outro valor de agrupamento")
            print("================================================")
    dados = dados[dados['Número Agrupado']==radar].reset_index(drop=True) #Seleciona apenas o radar que estamos testando
    
    #Recebe o horário divisor entre treino/teste
    horaok = False
    while not horaok:
        hora = input("Formato do horário esperado: 00:00:00 \n Qual o horário de corte entre treino/teste?: ")
        if len(hora)==8 and hora[2] == ":" and hora[5] == ":" and int(hora[0]+hora[1]) in range(0,25) and int(hora[3]+hora[4]) in range(0,60) and int(hora[6]+hora[7]) in range(0,60):
            horaok = True
            print("================================================")
        else:
            print("Input inválido, digite novamente o valor da hora")
            print("================================================")
    
    #CHECAR SE A SÉRIE É ESTACIONÁRIA
    diferenciacao = dados.Quantidade
    estacionaria = adfuller(diferenciacao)
    print('ADF Statistic: %f' % estacionaria[0],'| p-value: %f' % estacionaria[1])
    if estacionaria[1] < 0.05:
        print("A série do radar %s é estacionária, pois p-value < 0.05" %radar)
        pronto = True
        d = 0
    else:
        print("A série do radar %s não é estacionária, pois p-value > 0.05" %radar)
        pronto = False
        d = 0
        while not pronto:
            diferenciacao = diferenciacao.diff().dropna()
            d = d + 1
            estacionaria = adfuller(diferenciacao)
            print('ADF Statistic: %f' % estacionaria[0],'| p-value: %f' % estacionaria[1])                
            if estacionaria[1] < 0.05:
                pronto = True
    print("A ordem de diferenciação é %d" %d)
    
    diferenciar = input("Deseja modificar o valor de d? (S/N) ")
    if diferenciar =="S":
        d = int(input("Digite a ordem d desejada: "))
        for i in range(1,d+1):
            diferenciacao = diferenciacao.diff().dropna()
    
    #Plotando a série original e estacionária
    plt.rcParams.update({'figure.figsize':(9,7),'figure.dpi':120})
    fig,axis=plt.subplots(2,1,sharex=False)
    axis[0].plot(dados.Quantidade);axis[0].set_title('Série Original')
    axis[1].plot(diferenciacao);axis[1].set_title('Série Estacionária')
    plt.show()
    
    pronto = False
    while not pronto:
        #ORDEM DA AUTO REGRESSÃO (PACF nos dados estacionários)
        plt.rcParams.update({'figure.figsize':(9,7),'figure.dpi':120})
        fig,axis=plt.subplots(2,1,sharex=False)
        axis[0].set(ylim=(0,1.05))
        plot_pacf(diferenciacao,ax=axis[0],lags=100)
        
        #ORDEM DA MÉDIA MÓVEL (ACF nos dados estacionários)
        axis[1].set(ylim=(0,1.2))
        plot_acf(diferenciacao,ax=axis[1],lags=100)
        plt.show()
        
        p=int(input("Qual o valor de p adequado?: "))
        q=int(input("Qual o valor de q desejado?: "))
        
        #CONSTRUÇÃO DO MODELO
        #Separação em dados de treino e de teste
        dia_loc = datetime.datetime.strptime(str(lista_dias[posicao_dia])+" "+hora,'%Y-%m-%d %H:%M:%S') #Dia e hora escolhidos para separação treino/teste
        divisao = dados.loc[dados['Data']==dia_loc].index[0]
        previos = int(input("Quantos timelags anteriores utilizar na previsão? (Indicado acima de 1000, para entrar os dados dos dias anteriores) "))
        start = divisao - previos #Quantos steps prévios usar para a previsão
        # Separa o dataframe do radar escolhido em treino e validação
        treino = dados.loc[start:divisao]
        teste  = dados.loc[divisao:]
        
        steps= int(input("Quantos steps a frente das %s h deseja prever? " %hora))
        teste = teste[:steps+1]
        
        #Indexação (horários) da previsão
        data_prev = dados.loc[divisao+1:]
        data_prev = data_prev[:steps]
        
        #Instancia o modelo e faz o fit no treino
        model = ARIMA(treino['Quantidade'],order=(p,d,q))
        model_fit=model.fit(disp=-1)
        
        #Realiza o forecast
        fc,se,conf=model_fit.forecast(steps,alpha=0.05) #95% de confiança
        
        #Valores da previsão recebem o index da data_prev
        fc_series = pd.Series(fc,index=data_prev.index)
        lower_series = pd.Series(conf[:,0],index=data_prev.Data)
        upper_series = pd.Series(conf[:,1],index=data_prev.Data)
        
        
        start = divisao - 100 #Quantos steps prévios usar para plotar
        treino = dados.loc[start:divisao]
    
        #Plota a previsão do modelo:
        fig,ax=plt.subplots(figsize=(12,5))
        ax.plot(treino['Data'],treino['Quantidade'],label='Observações de fato, usadas para treino')
        ax.plot(teste['Data'],teste['Quantidade'],label='Comportamento de fato, após %s h' %hora)
        ax.plot(data_prev['Data'],fc_series,label='ARIMA')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.fill_between(lower_series.index,lower_series,upper_series,color='k',alpha=0.15)
        plt.title('Local: Radar %d' %radar)
        plt.legend(loc='upper left',fontsize=10)
        plt.show()
            
        #TABELA DOS DADOS PREVISTOS
        dados_prev = pd.Series(fc,index=data_prev.Data)
        dados_prev = dados_prev.to_frame()
        dados_prev.columns = ['Quantidade']
        dados_prev['Data'] = dados_prev.index
        dados_prev = dados_prev[['Data','Quantidade']]
        dados_prev = dados_prev.reset_index(drop=True)
#        print("Previsão:")
#        print(dados_prev)
#        print("")
        
        #TABELA DOS DADOS REAIS - ALTERAR PRA PEGAR O COMPORTAMENTO REAL
        dados_reais = teste[['Data','Quantidade']]
        dados_reais = dados_reais.reset_index(drop=True)
        dados_reais = dados_reais[1:]
        dados_reais = dados_reais.reset_index(drop=True)
#        print("Real:")
#        print(dados_reais)
#        print("")
        
        #CÁLCULO DO ERRO
        erros = list()
        for i in range(0,15):
            erro = dados_reais.loc[i,'Quantidade'] - dados_prev.loc[i,'Quantidade']
            erros.append(abs(erro))
        
        erros = pd.Series(erros, index=data_prev.Data)
        erros = erros.to_frame()
        erros.columns = ['Erro']
        erros['Data'] = erros.index
        erros = erros[['Data','Erro']]
        erros = erros.reset_index(drop=True)
        
#        print(erros)
        
        var_erro = statistics.variance(erros.Erro)
        print("")
        print("A variância do erro dessa simulação foi de: %f" %var_erro)
        print("================================================")
        print(model_fit.summary())
        print("================================================")
        
        #DEFINE QUAL SERIA O MODELO IDEAL PARA A PREVISÃO
        modell = pm.auto_arima(dados.Quantidade,start_p=0,star_q=0,test='adf',max_p=p,max_q=q,m=1,d=d,seasonal=False,start_P=0,D=0,trace=True,error_action='ignore',suppress_warnings=True,stepwise=True)     
        print(modell.summary())
        
        forecast_accuracy(fc, dados_reais.Quantidade)
        
        ok = input("O modelo está adequado? (S/N) ")
        if ok == "S":
            pronto = True
            print("Modelo definido para aplicação em outros horários")
            print("================================================")
        else:
            print("Escolha outras ordens de modelo")
            print("================================================")
    
    #SIMULAÇÃO DE OUTROS VALORES APÓS DEFINIÇÃO DO MODELO
    pronto = False
    ok = input("Deseja simular? (S/N): ")
    if ok == "N":
        pronto = True
        
    while not pronto:
        
        teste  = dados.loc[divisao:]
        steps= int(input("Quantos steps a frente das %s h deseja prever? " %hora ))
        print("================================================")
        teste = teste[:steps+1]
        data_prev = dados.loc[divisao+1:]
        data_prev = data_prev[:steps]
        
        #Realiza o forecast
        fc,se,conf=model_fit.forecast(steps,alpha=0.05) #95% de confiança
        
        #Valores da previsão recebem o index da data_prev
        fc_series = pd.Series(fc,index=data_prev.index)
        lower_series = pd.Series(conf[:,0],index=data_prev.Data)
        upper_series = pd.Series(conf[:,1],index=data_prev.Data)
        
        #Plota a previsão do modelo:
        fig,ax=plt.subplots(figsize=(12,5))
        ax.plot(treino['Data'],treino['Quantidade'],label='Observações de fato, usadas para treino')
        ax.plot(teste['Data'],teste['Quantidade'],label='Comportamento de fato, após %s h' %hora)
        ax.plot(data_prev['Data'],fc_series,label='ARIMA')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.fill_between(lower_series.index,lower_series,upper_series,color='k',alpha=0.15)
        plt.title('Local: Radar %d' %radar)
        plt.legend(loc='upper left',fontsize=10)
        plt.show()
        
        ok = input("Deseja fazer outra previsão? (S/N): ")
        if ok == "N":
            print("Simulação encerrada")
            pronto = True
        else:
            print("Iniciando outra simulação")
            print("================================================")
    
main()