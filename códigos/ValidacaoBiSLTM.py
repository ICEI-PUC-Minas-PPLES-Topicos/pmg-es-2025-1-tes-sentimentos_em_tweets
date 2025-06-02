# Essa parte é o codigo de REVALIDAÇÂO voce pode executar ela independente do treino DESDE QUE TENHA OS ARQUIVOS,
# se voce ta rodando um ambiente novo tem que descomentar os imports e rodar eles tbm
import pandas as pd
import numpy as np
import pickle
import re
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import os

# --- 0. Nomes dos Arquivos para Salvar/Carregar ---
MODEL_SAVE_FILE_BILSTM = 'bilstm_final_model.h5'
TOKENIZER_SAVE_FILE_BILSTM = 'bilstm_final_tokenizer.pickle'

# --- 1. Definições Globais e Constantes  ---
# Função de limpeza de texto
def clean_text(text): # Certifique-se que esta é a mesma função do treino
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Dicionário de sentimentos e número de classes
sentiment_map = {
    'Positive': 0, 'Negative': 1, 'Neutral': 2, 'Irrelevant': 3
}
sentiment_map_inv = {v: k for k, v in sentiment_map.items()}
num_classes = len(sentiment_map)

# Constantes do modelo BiLSTM (DEVEM SER AS MESMAS DO TREINAMENTO)
MAX_LEN = 150 # Crucial que seja o mesmo MAX_LEN do treinamento

# --- INÍCIO DA SEÇÃO DE CARREGAMENTO E REAVALIAÇÃO DO BiLSTM ---
# Esta seção tentará carregar um modelo e tokenizer previamente salvos para reavaliação.
print("\n\n### INÍCIO DA SEÇÃO DE CARREGAMENTO E REAVALIAÇÃO DO BiLSTM ###")

# Verificar se os arquivos salvos existem antes de tentar carregar
if os.path.exists(MODEL_SAVE_FILE_BILSTM) and os.path.exists(TOKENIZER_SAVE_FILE_BILSTM):
    print(f"\n--- Carregando Modelo BiLSTM de '{MODEL_SAVE_FILE_BILSTM}' para Reavaliação ---")
    # Adicione a importação de load_model se ainda não estiver no escopo
    from tensorflow.keras.models import load_model
    loaded_model_bilstm_reval = load_model(MODEL_SAVE_FILE_BILSTM)
    print("Modelo BiLSTM carregado com sucesso para reavaliação.")
    loaded_model_bilstm_reval.summary() # Opcional: para ver a estrutura do modelo carregado

    print(f"\n--- Carregando Tokenizer BiLSTM de '{TOKENIZER_SAVE_FILE_BILSTM}' para Reavaliação ---")
    with open(TOKENIZER_SAVE_FILE_BILSTM, 'rb') as handle:
        loaded_tokenizer_bilstm_reval = pickle.load(handle)
    print("Tokenizer BiLSTM carregado com sucesso para reavaliação.")

    # --- Re-carregar e Re-processar Dados de Validação ESPECIFICAMENTE para esta seção de Reavaliação ---
    print("\n--- Re-carregando e Re-processando Dados de Validação para BiLSTM (Reavaliação) ---")
    try:
        df_val_reval_load = pd.read_csv('twitter_validation.csv', header=None, names=['TweetID', 'Entity', 'Sentiment', 'TweetContent'])
    except FileNotFoundError:
        print("Arquivo 'twitter_validation.csv' não encontrado para reavaliação. Faça o upload.")
        df_val_reval_load = None

    if df_val_reval_load is not None:
        df_val_reval_load.dropna(subset=['TweetContent', 'Sentiment', 'Entity'], inplace=True)
        # Aplicar a limpeza de texto
        df_val_reval_load['Cleaned_Tweet'] = df_val_reval_load['TweetContent'].apply(clean_text)
        # Criar o texto de entrada combinado
        df_val_reval_load['Input_Text_BiLSTM'] = df_val_reval_load['Entity'].astype(str).str.lower() + " [sep] " + df_val_reval_load['Cleaned_Tweet']
        # Codificar os sentimentos
        df_val_reval_load['Sentiment_Encoded'] = df_val_reval_load['Sentiment'].map(sentiment_map)
        df_val_reval_load.dropna(subset=['Sentiment_Encoded'], inplace=True)

        # Preparar os dados para o modelo
        X_val_text_reval_load = df_val_reval_load['Input_Text_BiLSTM'].values
        y_val_reval_labels_load = df_val_reval_load['Sentiment_Encoded'].values.astype(int)

        # Usar o TOKENIZER CARREGADO para converter texto em sequências
        X_val_seq_reval_load = loaded_tokenizer_bilstm_reval.texts_to_sequences(X_val_text_reval_load)
        # Aplicar padding
        X_val_pad_reval_load = pad_sequences(X_val_seq_reval_load, maxlen=MAX_LEN, padding='post', truncating='post')
        # Converter rótulos para formato categórico
        y_val_cat_reval_load = to_categorical(y_val_reval_labels_load, num_classes=num_classes)

        print(f"Dados de validação para reavaliação processados. Shape X: {X_val_pad_reval_load.shape}, Shape Y: {y_val_cat_reval_load.shape}")

        # --- Reavaliação com Modelo Carregado ---
        print("\n--- Reavaliando o modelo BiLSTM carregado ---")
        # Usar o MODELO CARREGADO para avaliar
        loss_reval, accuracy_reval = loaded_model_bilstm_reval.evaluate(X_val_pad_reval_load, y_val_cat_reval_load, verbose=1)
        print(f"Perda (Loss) na validação (BiLSTM Reavaliação com Modelo Carregado): {loss_reval:.4f}")
        print(f"Acurácia na validação (BiLSTM Reavaliação com Modelo Carregado): {accuracy_reval:.4f}")

        y_pred_probs_reval_load = loaded_model_bilstm_reval.predict(X_val_pad_reval_load)
        y_pred_classes_reval_load = np.argmax(y_pred_probs_reval_load, axis=1)

        target_names_reval_load = [sentiment_map_inv[i] for i in range(num_classes)]
        print("\nRelatório de Classificação (BiLSTM Reavaliação com Modelo Carregado):")
        # Adicionar imports para classification_report e confusion_matrix se ainda não estiverem no escopo
        from sklearn.metrics import classification_report, confusion_matrix
        print(classification_report(y_val_reval_labels_load, y_pred_classes_reval_load, target_names=target_names_reval_load, zero_division=0))

        conf_matrix_reval_load = confusion_matrix(y_val_reval_labels_load, y_pred_classes_reval_load)
        # Adicionar imports para plt e sns se ainda não estiverem no escopo
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix_reval_load, annot=True, fmt='d', cmap='Blues', xticklabels=target_names_reval_load, yticklabels=target_names_reval_load)
        plt.xlabel('Predito'); plt.ylabel('Verdadeiro'); plt.title('Matriz de Confusão - BiLSTM (Reavaliação com Modelo Carregado)'); plt.show()
    else:
        print("Não foi possível carregar 'twitter_validation.csv' para a reavaliação do BiLSTM.")
else:
    print(f"Arquivos de modelo ('{MODEL_SAVE_FILE_BILSTM}') ou tokenizer ('{TOKENIZER_SAVE_FILE_BILSTM}') não encontrados para carregar.")
    print("Execute a seção de treinamento e salvamento do script completo primeiro, ou faça upload dos arquivos salvos se já os tiver.")

print("### FIM DA SEÇÃO DE CARREGAMENTO E REAVALIAÇÃO DO BiLSTM ###")