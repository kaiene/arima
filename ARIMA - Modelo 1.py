# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 17:10:28 2020

@author: Michelle e Kaiene

MODELO 1 = Modelo com dados de treino consecutivos, isto é, utiliza como base todos 
os dias anteriores da semana para a previsão

"""

import pandas as pd #Pacote para lidar com data frames
import numpy as np
import os
from IPython.display import display, Image #Pacote para lidar com figuras
import matplotlib.pyplot as plt #Pacote para lidar com gráficos
import statistics

from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.dates as mdates



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
    #Recebe o dia escolhido
    display(Image(filename='calendario.png')) #Colocar a imagem do calendário na mesma pasta do arquivo python
    diaok = False
    while not diaok:
        dia = int(input("Digite o número do dia desejado (1-30): "))
        if dia in range(1,31):
            diaok = True
        else:
            print("Input inválido, escolha outro valor de dia")
            print("================================================")
    
    data = padroniza_dia(dia) #Padroniza o valor do dia para um string de dois dígitos
    
    #Recebe o agrupamento temporal escolhido
    agrupamentook = False
    while not agrupamentook:
        print("Agrupamentos disponíveis: \n1Min \n2Min \n3Min \n4Min \n5Min \n")
        agrupamento = input("Digite o agrupamento temporal desejado: ")
        if agrupamento == "1Min" or agrupamento == "2Min" or agrupamento == "3Min" or agrupamento == "4Min" or agrupamento == "5Min":
            agrupamentook = True
        else:
            print("Input inválido, escolha outro valor de agrupamento")
            print("================================================")
            
    #Recebe o radar escolhido para previsão    
    radares = [10426, 10433, 10482, 10484, 10492, 10500, 10521, 10531]
    radarok = False
    while not radarok:
        print("Radares disponíveis:", radares, sep='\n' )
        radar = int(input("Qual o radar a ser previsto?: "))
        if radar in radares:
            radarok = True
        else:
            print("Input inválido, escolha outro valor de agrupamento")
            print("================================================")
    
    #Leitura do arquivo - Alterar com base na localização dos dados do seu computador
    path = "C:\\Users\\walmart\\Documents\\USP\\TCC\\Dados\\Dados Agrupados\\Grouped by frequency"
    dados = pd.read_csv(path + "\\"+ str(dia) +"\\" + str(dia) + "_group_" + str(agrupamento) + ".csv")
    dados.Data = pd.to_datetime(dados.Data) #Transforma a coluna Data em formato Data
    dados = dados[dados['Número Agrupado']==radar].reset_index(drop=True) #Seleciona apenas o radar que estamos testando
    
    #Recebe o horário divisor entre treino/teste
    horaok = False
    while not horaok:
        hora = input("Formato do horário esperado: 00:00:00 \n Qual o horário de corte entre treino/teste?: ")
        if len(hora)==8 and hora[2] == ":" and hora[5] == ":" and int(hora[0]+hora[1]) in range(0,25) and int(hora[3]+hora[4]) in range(0,60) and int(hora[6]+hora[7]) in range(0,60):
            horaok = True
        else:
            print("Input inválido, digite novamente o valor da hora")
            print("================================================")
            
    divisao = dados.loc[dados['Data']=='2018-03-'+ data + ' ' + str(hora)].index[0]
    start = divisao - 100 #Quantos steps prévios usar para a previsão
    # Separa o dataframe do radar escolhido em treino e validação
    treino = dados.loc[start:divisao]
    teste  = dados.loc[divisao:]
    
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
    
    plt.rcParams.update({'figure.figsize':(9,7),'figure.dpi':120})
    fig,axis=plt.subplots(2,1,sharex=False)
    axis[0].plot(dados.Quantidade);axis[0].set_title('Série Original')
    axis[1].plot(diferenciacao);axis[1].set_title('Série Estacionária')
    plt.show()
    
    pronto = False
    while not pronto:
        #ORDEM DA AUTO REGRESSÃO - PACF COM DADOS ESTACIONÁRIOS
        plt.rcParams.update({'figure.figsize':(9,7),'figure.dpi':120})
        fig,axis=plt.subplots(2,1,sharex=False)
        axis[0].set(ylim=(0,1.05))
        plot_pacf(diferenciacao,ax=axis[0],lags=100)
                
        #ORDEM DA MÉDIA MÓVEL - ACF COM DADOS ESTACIONÁRIOS
        axis[1].set(ylim=(0,1.2))
        plot_acf(diferenciacao,ax=axis[1],lags=100)
        plt.show()
        
        p=int(input("Qual o valor de p adequado?: "))
        q=int(input("Qual o valor de q desejado?: "))
        
        #CONSTRUÇÃO DO MODELO
        steps= int(input("Quantos steps a frente das %s h deseja prever? " %hora))
        teste = teste[:steps+1] #Vai do horário de corte (dado) até 15+1 steps
        
        #A previsão vai do horário de corte+1step por 15 steps
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
        #print("Previsão:")
        #print(dados_prev)
        #print("")
        
        #TABELA DOS DADOS REAIS
        dados_reais = teste[['Data','Quantidade']]
        dados_reais = dados_reais.reset_index(drop=True)
        dados_reais = dados_reais[1:]
        dados_reais = dados_reais.reset_index(drop=True)
        #print("Real:")
        #print(dados_reais)
        #print("")
        
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
        #print(erros)
        
        var_erro = statistics.variance(erros.Erro)
        print("")
        print("A variância do erro dessa simulação foi de: %f" %var_erro)
        print("================================================")
        print(model_fit.summary())
        print("================================================")
        
        forecast_accuracy(fc, dados_reais.Quantidade)
        
        #DEFINE QUAL SERIA O MODELO IDEAL PARA A PREVISÃO
        modell = pm.auto_arima(dados.Quantidade,start_p=0,star_q=0,test='adf',max_p=p,max_q=q,m=1,d=d,seasonal=False,start_P=0,D=0,trace=True,error_action='ignore',suppress_warnings=True,stepwise=True)     
        print(modell.summary())
        
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
    while not pronto:
        horaok = False
        while not horaok:
            hora = input("Formato do horário esperado: 00:00:00 \n Qual o horário de início da previsão?: ")
            if len(hora)==8 and hora[2] == ":" and hora[5] == ":" and int(hora[0]+hora[1]) in range(0,25) and int(hora[3]+hora[4]) in range(0,60) and int(hora[6]+hora[7]) in range(0,60):
                horaok = True
                print("================================================")
            else:
                print("Input inválido, digite novamente o valor da hora")
                print("================================================")
        
        divisao = dados.loc[dados['Data']=='2018-03-'+ data + ' ' + str(hora)].index[0]
        start = divisao - 100
        treino = dados.loc[start:divisao]
        teste  = dados.loc[divisao:]
        
        steps= int(input("Quantos steps a frente das %s h deseja prever?" %hora ))
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