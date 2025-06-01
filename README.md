# Análise Comparativa entre Modelos BiLSTM e BERT para Classificação de Sentimentos em Tweets

## Alunos integrantes da equipe

* Amanda Moura
* André Faria
* Luiz Gustavo Santos
* Pedro Ramos
* Philippe Vieira


## Professor responsável

* Leornado Vilela

## Instruções de Uso

Este guia descreve como carregar os modelos BiLSTM e BERT pré-treinados e executar a validação no dataset twitter_validation.csv usando o Google Colab.

Todos os arquivos necessarios de pré-treinamento e do dataset podem ser entrados neste [drive](https://drive.google.com/drive/folders/1vuXuz5NhDF-22B63aAhbfunEf9Y292pv?usp=drive_link)
(existem arquivos de mais de 400mb, por isso não podemos colocar diretamente no github).

Passo 0: Preparação do Ambiente e Upload dos Arquivos

Abra o notebook no [Google Colab](https://colab.research.google.com/drive/1mzZHgu_bzjSrEUB26N9QawpeJlvL8aUU?usp=sharing).

Faça o upload dos arquivos necessários para o ambiente do Colab. Você pode usar o ícone de pasta na barra lateral esquerda para abrir o gerenciador de arquivos e clicar no botão "Fazer upload":

Faça o upload de: twitter_validation.csv

Faça o upload de: bilstm_final_model.h5

Faça o upload de: bilstm_final_tokenizer.pickle

Execute o codigo de numero 2 que faz a validação do dataset no modelo BiSLTM

Para o modelo BERT:

Faça o upload de: twitter_validation.csv

Faça o uploand da pasta bert_final_model_saved/ .

Execute o codigo de numero 4 que faz a validação do dataset no modelo BERT



