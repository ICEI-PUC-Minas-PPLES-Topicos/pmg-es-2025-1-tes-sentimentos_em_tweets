# Esse script treina e valida o BiSLTM nao precisa rodar ele mais, so roda o de baixo que é o de validação com os arquivos necessarios dentro do ambiente do colab
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import os

# --- 0. Nomes dos Arquivos para Salvar/Carregar ---
MODEL_SAVE_FILE_BILSTM = 'bilstm_final_model.h5'
TOKENIZER_SAVE_FILE_BILSTM = 'bilstm_final_tokenizer.pickle'

# --- 1. Definições Globais e Constantes  ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

sentiment_map = {
    'Positive': 0, 'Negative': 1, 'Neutral': 2, 'Irrelevant': 3
}
sentiment_map_inv = {v: k for k, v in sentiment_map.items()}
num_classes = len(sentiment_map)

# Constantes do modelo BiLSTM (usadas no treino e na reavaliação ao reprocessar dados)
MAX_WORDS = 20000
MAX_LEN = 150
EMBEDDING_DIM = 128
BATCH_SIZE_TRAIN = 128 # Batch size para o treinamento
EPOCHS_TRAIN = 10    # Épocas para o treinamento

# Variáveis que serão definidas na seção de treino e usadas para salvar
# model_bilstm = None # Será o modelo Keras treinado
# tokenizer = None    # Será o Tokenizer Keras ajustado

print("### INÍCIO DA SEÇÃO DE TREINAMENTO E AVALIAÇÃO INICIAL DO BiLSTM ###")
print("Certifique-se de que as variáveis 'model_bilstm' e 'tokenizer' são definidas nesta seção.")


# --- 1. Carregamento dos Dados ---
print("Carregando dados para TREINAMENTO BiLSTM...")
try:
    df_train = pd.read_csv('twitter_training.csv', header=None, names=['TweetID', 'Entity', 'Sentiment', 'TweetContent'])
    df_val = pd.read_csv('twitter_validation.csv', header=None, names=['TweetID', 'Entity', 'Sentiment', 'TweetContent'])
except FileNotFoundError:
    print("Erro: Arquivos 'twitter_training.csv' ou 'twitter_validation.csv' não encontrados.")
    print("Por favor, faça o upload dos arquivos para o ambiente do Google Colab e tente novamente.")
    raise

print(f"Shape do treino: {df_train.shape}, Shape da validação: {df_val.shape}")

# --- 2. Pré-processamento Básico (para Treinamento) ---
print("Pré-processando dados para TREINAMENTO BiLSTM...")
df_train.dropna(subset=['TweetContent', 'Sentiment', 'Entity'], inplace=True)
df_val.dropna(subset=['TweetContent', 'Sentiment', 'Entity'], inplace=True)
df_train['Cleaned_Tweet'] = df_train['TweetContent'].apply(clean_text)
df_val['Cleaned_Tweet'] = df_val['TweetContent'].apply(clean_text)
df_train['Input_Text_BiLSTM'] = df_train['Entity'].astype(str).str.lower() + " [sep] " + df_train['Cleaned_Tweet']
df_val['Input_Text_BiLSTM'] = df_val['Entity'].astype(str).str.lower() + " [sep] " + df_val['Cleaned_Tweet']
df_train['Sentiment_Encoded'] = df_train['Sentiment'].map(sentiment_map)
df_val['Sentiment_Encoded'] = df_val['Sentiment'].map(sentiment_map)
df_train.dropna(subset=['Sentiment_Encoded'], inplace=True)
df_val.dropna(subset=['Sentiment_Encoded'], inplace=True)

X_train = df_train['Input_Text_BiLSTM'].values
y_train = df_train['Sentiment_Encoded'].values
X_val = df_val['Input_Text_BiLSTM'].values # Usado para X_val_pad
y_val = df_val['Sentiment_Encoded'].values # Usado para relatório de classificação

y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_val_cat = to_categorical(y_val, num_classes=num_classes)

# --- 3. Tokenização e Padding (para Treinamento) ---
print("Tokenizando e aplicando padding para TREINAMENTO BiLSTM...")
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<unk>") # tokenizer é definido AQUI
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val) # X_val é usado aqui
X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post', truncating='post')
X_val_pad = pad_sequences(X_val_seq, maxlen=MAX_LEN, padding='post', truncating='post') # X_val_pad é definido AQUI

# --- 4. Construção do Modelo BiLSTM ---
print("Construindo modelo BiLSTM...")
model_bilstm = Sequential([ # model_bilstm é definido AQUI
    Embedding(input_dim=MAX_WORDS, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),
    SpatialDropout1D(0.3),
    Bidirectional(LSTM(units=128, return_sequences=True, recurrent_dropout=0.3)),
    Bidirectional(LSTM(units=64, recurrent_dropout=0.2)),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
model_bilstm.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_bilstm.summary()

# --- 5. Treinamento do Modelo BiLSTM ---
print("\nIniciando o TREINAMENTO do modelo BiLSTM...")
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001, verbose=1)
history_bilstm = model_bilstm.fit( # history_bilstm é definido AQUI
    X_train_pad, y_train_cat,
    epochs=EPOCHS_TRAIN, batch_size=BATCH_SIZE_TRAIN,
    validation_data=(X_val_pad, y_val_cat), # Usa X_val_pad, y_val_cat
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)
print("Treinamento do BiLSTM concluído.")

# --- 6. Avaliação Inicial do Modelo BiLSTM (após o treino) ---
print("\nAvaliando o modelo BiLSTM treinado no conjunto de validação...")
loss, accuracy = model_bilstm.evaluate(X_val_pad, y_val_cat, verbose=0) # Usa X_val_pad, y_val_cat
print(f"Perda (Loss) na validação: {loss:.4f}")
print(f"Acurácia na validação: {accuracy:.4f}")
y_pred_probs_bilstm = model_bilstm.predict(X_val_pad) # Usa X_val_pad
y_pred_classes_bilstm = np.argmax(y_pred_probs_bilstm, axis=1)
target_names = [sentiment_map_inv[i] for i in range(num_classes)]
print("\nRelatório de Classificação (BiLSTM - Avaliação Inicial):")
print(classification_report(y_val, y_pred_classes_bilstm, target_names=target_names, zero_division=0)) # Usa y_val
conf_matrix = confusion_matrix(y_val, y_pred_classes_bilstm) # Usa y_val
plt.figure(figsize=(8, 6)); sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title('Matriz de Confusão - BiLSTM (Avaliação Inicial)'); plt.xlabel('Predito'); plt.ylabel('Verdadeiro'); plt.show()

def plot_training_history(history, model_name): # Função de plotar
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1); plt.plot(history.history['accuracy'], label='Acurácia de Treino'); plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
    plt.title(f'Acurácia - {model_name}'); plt.xlabel('Época'); plt.ylabel('Acurácia'); plt.legend()
    plt.subplot(1, 2, 2); plt.plot(history.history['loss'], label='Perda de Treino'); plt.plot(history.history['val_loss'], label='Perda de Validação')
    plt.title(f'Perda (Loss) - {model_name}'); plt.xlabel('Época'); plt.ylabel('Perda'); plt.legend()
    plt.tight_layout(); plt.show()
plot_training_history(history_bilstm, "BiLSTM (Avaliação Inicial)")

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
print("### FIM DA SEÇÃO DE TREINAMENTO E AVALIAÇÃO INICIAL DO BiLSTM ###\n")


# --- 7. SALVAR O MODELO E O TOKENIZER (APÓS TREINAMENTO E AVALIAÇÃO INICIAL) ---
# Esta seção só será útil se a seção de treinamento acima foi executada e 'model_bilstm' e 'tokenizer' existem.
if 'model_bilstm' in locals() and 'tokenizer' in locals():
    print(f"--- Salvando o Modelo BiLSTM em '{MODEL_SAVE_FILE_BILSTM}' ---")
    model_bilstm.save(MODEL_SAVE_FILE_BILSTM)
    print("Modelo BiLSTM salvo com sucesso.")

    print(f"\n--- Salvando o Tokenizer BiLSTM em '{TOKENIZER_SAVE_FILE_BILSTM}' ---")
    with open(TOKENIZER_SAVE_FILE_BILSTM, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Tokenizer BiLSTM salvo com sucesso.")
else:
    print("AVISO: 'model_bilstm' ou 'tokenizer' não definidos na seção de treino. O salvamento não será executado.")
    print("Certifique-se de que o código de treinamento completo foi colado e executado acima.")


# --- 8. CARREGAR MODELO BiLSTM SALVO E REAVALIAR  ---
print("\n\n### INÍCIO DA SEÇÃO DE CARREGAMENTO E REAVALIAÇÃO DO BiLSTM ###")
print("Esta seção tentará carregar um modelo e tokenizer previamente salvos para reavaliação.")

# Verificar se os arquivos salvos existem antes de tentar carregar
if os.path.exists(MODEL_SAVE_FILE_BILSTM) and os.path.exists(TOKENIZER_SAVE_FILE_BILSTM):
    print(f"\n--- Carregando Modelo BiLSTM de '{MODEL_SAVE_FILE_BILSTM}' para Reavaliação ---")
    loaded_model_bilstm_reval = load_model(MODEL_SAVE_FILE_BILSTM)
    print("Modelo BiLSTM carregado com sucesso para reavaliação.")

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
        # Considerar sair ou pular esta seção se o arquivo não for encontrado
        df_val_reval_load = None

    if df_val_reval_load is not None:
        df_val_reval_load.dropna(subset=['TweetContent', 'Sentiment', 'Entity'], inplace=True)
        df_val_reval_load['Cleaned_Tweet'] = df_val_reval_load['TweetContent'].apply(clean_text)
        df_val_reval_load['Input_Text_BiLSTM'] = df_val_reval_load['Entity'].astype(str).str.lower() + " [sep] " + df_val_reval_load['Cleaned_Tweet']
        df_val_reval_load['Sentiment_Encoded'] = df_val_reval_load['Sentiment'].map(sentiment_map)
        df_val_reval_load.dropna(subset=['Sentiment_Encoded'], inplace=True)

        X_val_text_reval_load = df_val_reval_load['Input_Text_BiLSTM'].values
        y_val_reval_labels_load = df_val_reval_load['Sentiment_Encoded'].values.astype(int)

        X_val_seq_reval_load = loaded_tokenizer_bilstm_reval.texts_to_sequences(X_val_text_reval_load)
        X_val_pad_reval_load = pad_sequences(X_val_seq_reval_load, maxlen=MAX_LEN, padding='post', truncating='post')
        y_val_cat_reval_load = to_categorical(y_val_reval_labels_load, num_classes=num_classes)

        print(f"Dados de validação para reavaliação processados. Shape: {X_val_pad_reval_load.shape}")

        # --- Reavaliação com Modelo Carregado ---
        print("\n--- Reavaliando o modelo BiLSTM carregado ---")
        loss_reval, accuracy_reval = loaded_model_bilstm_reval.evaluate(X_val_pad_reval_load, y_val_cat_reval_load, verbose=1)
        print(f"Perda (Loss) na validação (BiLSTM Reavaliação com Modelo Carregado): {loss_reval:.4f}")
        print(f"Acurácia na validação (BiLSTM Reavaliação com Modelo Carregado): {accuracy_reval:.4f}")

        y_pred_probs_reval_load = loaded_model_bilstm_reval.predict(X_val_pad_reval_load)
        y_pred_classes_reval_load = np.argmax(y_pred_probs_reval_load, axis=1)

        target_names_reval_load = [sentiment_map_inv[i] for i in range(num_classes)]
        print("\nRelatório de Classificação (BiLSTM Reavaliação com Modelo Carregado):")
        print(classification_report(y_val_reval_labels_load, y_pred_classes_reval_load, target_names=target_names_reval_load, zero_division=0))

        conf_matrix_reval_load = confusion_matrix(y_val_reval_labels_load, y_pred_classes_reval_load)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix_reval_load, annot=True, fmt='d', cmap='Blues', xticklabels=target_names_reval_load, yticklabels=target_names_reval_load)
        plt.xlabel('Predito'); plt.ylabel('Verdadeiro'); plt.title('Matriz de Confusão - BiLSTM (Reavaliação com Modelo Carregado)'); plt.show()
    else:
        print("Não foi possível carregar 'twitter_validation.csv' para a reavaliação do BiLSTM.")
else:
    print(f"Arquivos de modelo ('{MODEL_SAVE_FILE_BILSTM}') ou tokenizer ('{TOKENIZER_SAVE_FILE_BILSTM}') não encontrados para carregar.")
    print("Execute a seção de treinamento e salvamento primeiro, ou faça upload dos arquivos salvos se já os tiver.")

print("### FIM DA SEÇÃO DE CARREGAMENTO E REAVALIAÇÃO DO BiLSTM ###")