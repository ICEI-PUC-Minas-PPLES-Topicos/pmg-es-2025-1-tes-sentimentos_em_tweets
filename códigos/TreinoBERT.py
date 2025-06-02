# Esse script treina e valida o BERT nao precisa rodar ele mais, so roda o de baixo que é o de validação com os arquivos necessarios dentro do ambiente do colab
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
import os

# --- 0. Nomes dos Arquivos/Diretórios para Salvar/Carregar ---
MODEL_SAVE_DIR_BERT = './bert_final_model_saved/' # Diretório para salvar/carregar o modelo BERT

# --- 1. Definições Globais e Constantes (Serão usadas em ambas as seções) ---
def clean_text_for_bert(text):
    text = str(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def bert_encode(entities, tweets, tokenizer, max_len):
    input_ids, attention_masks, token_type_ids_list = [], [], []
    for entity, tweet in zip(entities, tweets):
        encoded_dict = tokenizer.encode_plus(
            entity, tweet, add_special_tokens=True, max_length=max_len,
            padding='max_length', truncation=True, return_attention_mask=True,
            return_token_type_ids=True, return_tensors='pt',
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        token_type_ids_list.append(encoded_dict['token_type_ids'])
    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0), torch.cat(token_type_ids_list, dim=0)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

sentiment_map = {
    'Positive': 0, 'Negative': 1, 'Neutral': 2, 'Irrelevant': 3
}
sentiment_map_inv = {v: k for k, v in sentiment_map.items()}
num_classes = len(sentiment_map)

MODEL_NAME_BERT_PRETRAINED = 'bert-base-uncased'
MAX_LEN_BERT = 128
BATCH_SIZE_BERT_TRAIN = 32 # Batch size para treinamento e DataLoaders
EPOCHS_BERT_TRAIN = 3

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Usando GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("GPU não encontrada, usando CPU.")

# Variáveis que serão definidas na seção de treino e usadas para salvar
# model_bert = None # Será o modelo BERT treinado
# tokenizer_bert = None # Será o BertTokenizer carregado/ajustado

print("### INÍCIO DA SEÇÃO DE TREINAMENTO E AVALIAÇÃO INICIAL DO BERT ###")
print("Certifique-se de que as variáveis 'model_bert' e 'tokenizer' (para BERT) são definidas nesta seção.")

# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
# >>> Substitua este comentário e as linhas de exemplo abaixo pelo seu CÓDIGO ORIGINAL COMPLETO <<<
# >>> de carregamento de dados, pré-processamento, definição, treinamento e avaliação inicial do BERT. <<<
# >>> As variáveis `model_bert` e `tokenizer` (BERT) DEVEM ser definidas aqui.               <<<
# >>> As variáveis `val_dataloader`, `training_stats` (ou `df_stats`) também são esperadas  <<<
# >>> pela avaliação inicial e plotagem de histórico.                                        <<<

# Exemplo de estrutura do que colar aqui:
# --- 1. Carregamento dos Dados ---
print("Carregando dados para TREINAMENTO BERT...")
try:
    df_train = pd.read_csv('twitter_training.csv', header=None, names=['TweetID', 'Entity', 'Sentiment', 'TweetContent'])
    df_val = pd.read_csv('twitter_validation.csv', header=None, names=['TweetID', 'Entity', 'Sentiment', 'TweetContent'])
except FileNotFoundError:
    print("Erro: Arquivos 'twitter_training.csv' ou 'twitter_validation.csv' não encontrados.")
    exit()
print(f"Shape do treino: {df_train.shape}, Shape da validação: {df_val.shape}")

# --- 2. Pré-processamento (para Treinamento) ---
print("Pré-processando dados para TREINAMENTO BERT...")
df_train.dropna(subset=['TweetContent', 'Sentiment', 'Entity'], inplace=True)
df_val.dropna(subset=['TweetContent', 'Sentiment', 'Entity'], inplace=True)
df_train['TweetContent_Cleaned'] = df_train['TweetContent'].apply(clean_text_for_bert)
df_val['TweetContent_Cleaned'] = df_val['TweetContent'].apply(clean_text_for_bert)
train_texts_entity = df_train['Entity'].astype(str).tolist()
train_texts_tweet = df_train['TweetContent_Cleaned'].astype(str).tolist()
val_texts_entity = df_val['Entity'].astype(str).tolist() # Usado para val_dataloader
val_texts_tweet = df_val['TweetContent_Cleaned'].astype(str).tolist() # Usado para val_dataloader
df_train['Sentiment_Encoded'] = df_train['Sentiment'].map(sentiment_map)
df_val['Sentiment_Encoded'] = df_val['Sentiment'].map(sentiment_map)
df_train.dropna(subset=['Sentiment_Encoded'], inplace=True)
df_val.dropna(subset=['Sentiment_Encoded'], inplace=True)
# Reajustar listas se NaNs em Sentiment_Encoded foram dropados
train_texts_entity = df_train['Entity'].astype(str).tolist() # Atualiza após dropna
train_texts_tweet = df_train['TweetContent_Cleaned'].astype(str).tolist() # Atualiza após dropna
val_texts_entity = df_val['Entity'].astype(str).tolist() # Atualiza após dropna
val_texts_tweet = df_val['TweetContent_Cleaned'].astype(str).tolist() # Atualiza após dropna
train_labels = df_train['Sentiment_Encoded'].values.astype(int)
val_labels = df_val['Sentiment_Encoded'].values.astype(int) # Usado para val_dataloader

# --- 3. Tokenização para BERT (para Treinamento) ---
print("Tokenizando para TREINAMENTO BERT...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME_BERT_PRETRAINED) # tokenizer (BERT) é definido AQUI
train_input_ids, train_attention_masks, train_token_type_ids = bert_encode(train_texts_entity, train_texts_tweet, tokenizer, MAX_LEN_BERT)
val_input_ids, val_attention_masks, val_token_type_ids = bert_encode(val_texts_entity, val_texts_tweet, tokenizer, MAX_LEN_BERT) # val_input_ids etc. são definidos AQUI
train_labels_tensor = torch.tensor(train_labels)
val_labels_tensor = torch.tensor(val_labels) # val_labels_tensor é definido AQUI

# --- 4. Preparar DataLoaders (para Treinamento) ---
print("Preparando DataLoaders para TREINAMENTO BERT...")
train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_token_type_ids, train_labels_tensor)
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE_BERT_TRAIN)
val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_token_type_ids, val_labels_tensor) # Usa val_input_ids, val_attention_masks, val_token_type_ids, val_labels_tensor
val_sampler = SequentialSampler(val_dataset)
val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=BATCH_SIZE_BERT_TRAIN) # val_dataloader é definido AQUI

# --- 5. Construção do Modelo BERT ---
print(f"Construindo modelo BERT ({MODEL_NAME_BERT_PRETRAINED})...")
model_bert = BertForSequenceClassification.from_pretrained( # model_bert é definido AQUI
    MODEL_NAME_BERT_PRETRAINED, num_labels=num_classes,
    output_attentions=False, output_hidden_states=False,
)
model_bert.to(device)

# --- 6. Treinamento do Modelo BERT ---
print("\nIniciando o TREINAMENTO do modelo BERT...")
optimizer = AdamW(model_bert.parameters(), lr=2e-5, eps=1e-8)
total_steps = len(train_dataloader) * EPOCHS_BERT_TRAIN
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
training_stats = [] # training_stats é definido AQUI

for epoch_i in range(0, EPOCHS_BERT_TRAIN):
    print(f"\n======== Época {epoch_i + 1} / {EPOCHS_BERT_TRAIN} ========")
    total_train_loss = 0
    model_bert.train()
    for step, batch in enumerate(train_dataloader):
        if step % 50 == 0 and not step == 0: print(f'  Batch {step} de {len(train_dataloader)}.')
        b_input_ids,b_input_mask,b_token_type_ids,b_labels = batch[0].to(device),batch[1].to(device),batch[2].to(device),batch[3].to(device)
        model_bert.zero_grad()
        outputs = model_bert(b_input_ids, token_type_ids=b_token_type_ids, attention_mask=b_input_mask, labels=b_labels)
        loss, logits = outputs.loss, outputs.logits
        total_train_loss += loss.item()
        loss.backward(); torch.nn.utils.clip_grad_norm_(model_bert.parameters(),1.0); optimizer.step(); scheduler.step()
    avg_train_loss = total_train_loss / len(train_dataloader)
    print(f"  Perda (Loss) média de treino: {avg_train_loss:.4f}")
    print("\nRodando Validação (durante treino)...")
    model_bert.eval()
    total_eval_accuracy, total_eval_loss, all_preds_epoch, all_true_labels_epoch = 0, 0, [], []
    for batch in val_dataloader: # Usa val_dataloader
        b_input_ids,b_input_mask,b_token_type_ids,b_labels = batch[0].to(device),batch[1].to(device),batch[2].to(device),batch[3].to(device)
        with torch.no_grad(): outputs = model_bert(b_input_ids,token_type_ids=b_token_type_ids,attention_mask=b_input_mask,labels=b_labels)
        loss, logits = outputs.loss, outputs.logits
        total_eval_loss += loss.item()
        detached_logits, detached_labels = logits.detach().cpu().numpy(), b_labels.to('cpu').numpy()
        total_eval_accuracy += flat_accuracy(detached_logits, detached_labels)
        all_preds_epoch.extend(np.argmax(detached_logits, axis=1).flatten()) # Coleta preds para avaliação final da ÉPOCA
        all_true_labels_epoch.extend(detached_labels.flatten()) # Coleta labels para avaliação final da ÉPOCA
    avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
    avg_val_loss = total_eval_loss / len(val_dataloader)
    print(f"  Acurácia na validação: {avg_val_accuracy:.4f}")
    print(f"  Perda (Loss) na validação: {avg_val_loss:.4f}")
    training_stats.append({'epoch':epoch_i + 1,'Training Loss':avg_train_loss,'Valid. Loss':avg_val_loss,'Valid. Accur.':avg_val_accuracy})
    # Guardar as predições da última época para o relatório final
    if epoch_i == EPOCHS_BERT_TRAIN - 1:
        all_preds_final_train_run = all_preds_epoch
        all_true_labels_final_train_run = all_true_labels_epoch

print("\nTreinamento do BERT concluído!")

# --- 7. Avaliação Final do Modelo BERT (após o treino) ---
print("\n--- Avaliação Final do Modelo BERT (após treino completo) ---")
y_pred_classes_bert_train_run = np.array(all_preds_final_train_run)
y_val_bert_train_run = np.array(all_true_labels_final_train_run)
target_names_bert_train_run = [sentiment_map_inv[i] for i in range(num_classes)]
print("\nRelatório de Classificação (BERT - Avaliação Inicial):")
print(classification_report(y_val_bert_train_run, y_pred_classes_bert_train_run, target_names=target_names_bert_train_run, zero_division=0))
conf_matrix_bert_train_run = confusion_matrix(y_val_bert_train_run, y_pred_classes_bert_train_run)
plt.figure(figsize=(8,6)); sns.heatmap(conf_matrix_bert_train_run, annot=True, fmt='d', cmap='Greens', xticklabels=target_names_bert_train_run, yticklabels=target_names_bert_train_run)
plt.title('Matriz de Confusão - BERT (Avaliação Inicial)'); plt.xlabel('Predito'); plt.ylabel('Verdadeiro'); plt.show()
df_stats = pd.DataFrame(data=training_stats); df_stats = df_stats.set_index('epoch') # df_stats é definido AQUI
plt.figure(figsize=(12,4)); plt.subplot(1,2,1); plt.plot(df_stats['Training Loss'], label="Treino"); plt.plot(df_stats['Valid. Loss'], label="Validação")
plt.title("Perda de Treino & Validação - BERT"); plt.xlabel("Época"); plt.ylabel("Loss"); plt.legend();
plt.subplot(1,2,2); plt.plot(df_stats['Valid. Accur.'], label="Validação"); plt.title("Acurácia de Validação - BERT")
plt.xlabel("Época"); plt.ylabel("Acurácia"); plt.legend(); plt.tight_layout(); plt.show()

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
print("### FIM DA SEÇÃO DE TREINAMENTO E AVALIAÇÃO INICIAL DO BERT ###\n")


# --- 8. SALVAR O MODELO E O TOKENIZER (APÓS TREINAMENTO E AVALIAÇÃO INICIAL) ---
if 'model_bert' in locals() and 'tokenizer' in locals():
    if not os.path.exists(MODEL_SAVE_DIR_BERT):
        os.makedirs(MODEL_SAVE_DIR_BERT)
    print(f"\n--- Salvando o Modelo e Tokenizer BERT em '{MODEL_SAVE_DIR_BERT}' ---")
    model_bert.save_pretrained(MODEL_SAVE_DIR_BERT)
    tokenizer.save_pretrained(MODEL_SAVE_DIR_BERT) # tokenizer aqui é o BertTokenizer
    print("Modelo e Tokenizer BERT salvos com sucesso.")
else:
    print("AVISO: 'model_bert' ou 'tokenizer' (BERT) não definidos na seção de treino. O salvamento não será executado.")
    print("Certifique-se de que o código de treinamento completo foi colado e executado acima.")


# --- 9. CARREGAR MODELO BERT SALVO E REAVALIAR (PODE SER EXECUTADO INDEPENDENTEMENTE SE O DIRETÓRIO EXISTIR) ---
print("\n\n### INÍCIO DA SEÇÃO DE CARREGAMENTO E REAVALIAÇÃO DO BERT ###")
print("Esta seção tentará carregar um modelo e tokenizer BERT previamente salvos para reavaliação.")

if os.path.exists(MODEL_SAVE_DIR_BERT):
    print(f"\n--- Carregando Modelo BERT de '{MODEL_SAVE_DIR_BERT}' para Reavaliação ---")
    loaded_model_bert_reval = BertForSequenceClassification.from_pretrained(MODEL_SAVE_DIR_BERT)
    loaded_tokenizer_bert_reval = BertTokenizer.from_pretrained(MODEL_SAVE_DIR_BERT)
    loaded_model_bert_reval.to(device)
    print("Modelo e Tokenizer BERT carregados com sucesso para reavaliação.")

    # --- Re-carregar e Re-processar Dados de Validação ESPECIFICAMENTE para esta seção de Reavaliação ---
    print("\n--- Re-carregando e Re-processando Dados de Validação para BERT (Reavaliação) ---")
    try:
        df_val_reval_load = pd.read_csv('twitter_validation.csv', header=None, names=['TweetID', 'Entity', 'Sentiment', 'TweetContent'])
    except FileNotFoundError:
        print("Arquivo 'twitter_validation.csv' não encontrado para reavaliação. Faça o upload.")
        df_val_reval_load = None

    if df_val_reval_load is not None:
        df_val_reval_load.dropna(subset=['TweetContent', 'Sentiment', 'Entity'], inplace=True)
        df_val_reval_load['TweetContent_Cleaned'] = df_val_reval_load['TweetContent'].apply(clean_text_for_bert)
        val_texts_entity_reval_load = df_val_reval_load['Entity'].astype(str).tolist()
        val_texts_tweet_reval_load = df_val_reval_load['TweetContent_Cleaned'].astype(str).tolist()
        df_val_reval_load['Sentiment_Encoded'] = df_val_reval_load['Sentiment'].map(sentiment_map)
        df_val_reval_load.dropna(subset=['Sentiment_Encoded'], inplace=True)
        val_labels_reval_load = df_val_reval_load['Sentiment_Encoded'].values.astype(int)

        val_input_ids_reval_load, val_attention_masks_reval_load, val_token_type_ids_reval_load = bert_encode(
            val_texts_entity_reval_load, val_texts_tweet_reval_load, loaded_tokenizer_bert_reval, MAX_LEN_BERT
        )
        val_labels_tensor_reval_load = torch.tensor(val_labels_reval_load)

        val_dataset_reval_load = TensorDataset(val_input_ids_reval_load, val_attention_masks_reval_load, val_token_type_ids_reval_load, val_labels_tensor_reval_load)
        val_sampler_reval_load = SequentialSampler(val_dataset_reval_load)
        # Usar BATCH_SIZE_BERT_TRAIN para consistência, ou defina um BATCH_SIZE_BERT_EVAL se preferir
        val_dataloader_reval_load = DataLoader(val_dataset_reval_load, sampler=val_sampler_reval_load, batch_size=BATCH_SIZE_BERT_TRAIN)

        print(f"Dados de validação para reavaliação BERT processados. Número de batches: {len(val_dataloader_reval_load)}")

        # --- Reavaliação com Modelo Carregado ---
        print("\n--- Reavaliando o modelo BERT carregado ---")
        loaded_model_bert_reval.eval()
        all_preds_bert_reval_final, all_true_labels_bert_reval_final = [], []
        total_eval_loss_bert_reval_final, total_eval_accuracy_bert_reval_final = 0,0

        for batch in val_dataloader_reval_load:
            b_input_ids,b_input_mask,b_token_type_ids,b_labels = batch[0].to(device),batch[1].to(device),batch[2].to(device),batch[3].to(device)
            with torch.no_grad(): outputs = loaded_model_bert_reval(b_input_ids,token_type_ids=b_token_type_ids,attention_mask=b_input_mask,labels=b_labels)
            loss, logits = outputs.loss, outputs.logits
            total_eval_loss_bert_reval_final += loss.item()
            detached_logits, detached_labels = logits.detach().cpu().numpy(), b_labels.to('cpu').numpy()
            total_eval_accuracy_bert_reval_final += flat_accuracy(detached_logits, detached_labels)
            all_preds_bert_reval_final.extend(np.argmax(detached_logits, axis=1).flatten())
            all_true_labels_bert_reval_final.extend(detached_labels.flatten())

        avg_val_accuracy_reval = total_eval_accuracy_bert_reval_final / len(val_dataloader_reval_load)
        avg_val_loss_reval = total_eval_loss_bert_reval_final / len(val_dataloader_reval_load)
        print(f"  Acurácia na validação (BERT Reavaliação com Modelo Carregado): {avg_val_accuracy_reval:.4f}")
        print(f"  Perda (Loss) na validação (BERT Reavaliação com Modelo Carregado): {avg_val_loss_reval:.4f}")

        y_pred_classes_bert_reval_final = np.array(all_preds_bert_reval_final)
        y_val_bert_reval_final = np.array(all_true_labels_bert_reval_final)
        target_names_bert_reval = [sentiment_map_inv[i] for i in range(num_classes)]
        print("\nRelatório de Classificação (BERT Reavaliação com Modelo Carregado):")
        print(classification_report(y_val_bert_reval_final, y_pred_classes_bert_reval_final, target_names=target_names_bert_reval, zero_division=0))
        conf_matrix_bert_reval = confusion_matrix(y_val_bert_reval_final, y_pred_classes_bert_reval_final)
        plt.figure(figsize=(8,6)); sns.heatmap(conf_matrix_bert_reval, annot=True, fmt='d', cmap='Greens', xticklabels=target_names_bert_reval, yticklabels=target_names_bert_reval)
        plt.title('Matriz de Confusão - BERT (Reavaliação com Modelo Carregado)'); plt.xlabel('Predito'); plt.ylabel('Verdadeiro'); plt.show()
    else:
        print("Não foi possível carregar 'twitter_validation.csv' para a reavaliação do BERT.")
else:
    print(f"Diretório do modelo BERT salvo ('{MODEL_SAVE_DIR_BERT}') não encontrado para carregar.")
    print("Execute a seção de treinamento e salvamento primeiro, ou faça upload do diretório salvo se já o tiver.")

print("### FIM DA SEÇÃO DE CARREGAMENTO E REAVALIAÇÃO DO BERT ###")