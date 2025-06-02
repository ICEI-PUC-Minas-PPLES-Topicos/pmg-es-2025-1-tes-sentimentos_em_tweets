# Essa parte é o codigo de REVALIDAÇÂO voce pode executar ela independente do treino DESDE QUE TENHA OS ARQUIVOS,
# se voce ta rodando um ambiente novo tem que descomentar os imports e rodar eles tbm
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification
import os

# --- 0. Nomes dos Arquivos/Diretórios para Salvar/Carregar (COMO DEFINIDO NO SCRIPT ANTERIOR) ---
MODEL_SAVE_DIR_BERT = './bert_final_model_saved/' # Diretório onde o modelo BERT foi salvo

# --- 1. Definições Globais e Constantes (COMO DEFINIDO NO SCRIPT ANTERIOR) ---
# Função de limpeza de texto para BERT
def clean_text_for_bert(text): # Certifique-se que esta é a mesma função do treino
    text = str(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Função de tokenização BERT
def bert_encode(entities, tweets, tokenizer, max_len): # Certifique-se que esta é a mesma função do treino
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

# Função para calcular acurácia
def flat_accuracy(preds, labels): # Certifique-se que esta é a mesma função do treino
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# Dicionário de sentimentos e número de classes
sentiment_map = {
    'Positive': 0, 'Negative': 1, 'Neutral': 2, 'Irrelevant': 3
}
sentiment_map_inv = {v: k for k, v in sentiment_map.items()}
num_classes = len(sentiment_map)

# Constantes do modelo BERT (DEVEM SER AS MESMAS DO TREINAMENTO)
MAX_LEN_BERT = 128
BATCH_SIZE_BERT_EVAL = 32 # Pode ser o mesmo BATCH_SIZE_BERT_TRAIN ou diferente para avaliação

# Configurar dispositivo
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Usando GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("GPU não encontrada, usando CPU.")


# --- INÍCIO DA SEÇÃO DE CARREGAMENTO E REAVALIAÇÃO DO BERT ---
# Esta seção tentará carregar um modelo e tokenizer BERT previamente salvos para reavaliação.
print("\n\n### INÍCIO DA SEÇÃO DE CARREGAMENTO E REAVALIAÇÃO DO BERT ###")

# Verificar se o diretório salvo existe e contém arquivos chave
# (você pode adicionar verificações mais robustas para os arquivos específicos se desejar)
if os.path.exists(MODEL_SAVE_DIR_BERT) and \
   os.path.exists(os.path.join(MODEL_SAVE_DIR_BERT, "config.json")) and \
   os.path.exists(os.path.join(MODEL_SAVE_DIR_BERT, "vocab.txt")):
    print(f"\n--- Carregando Modelo BERT de '{MODEL_SAVE_DIR_BERT}' para Reavaliação ---")
    # Adicionar imports para BertForSequenceClassification e BertTokenizer se não estiverem no escopo
    from transformers import BertTokenizer, BertForSequenceClassification
    loaded_model_bert_reval = BertForSequenceClassification.from_pretrained(MODEL_SAVE_DIR_BERT)
    loaded_tokenizer_bert_reval = BertTokenizer.from_pretrained(MODEL_SAVE_DIR_BERT)
    loaded_model_bert_reval.to(device) # Mover modelo para o dispositivo
    print("Modelo e Tokenizer BERT carregados com sucesso para reavaliação.")
    # loaded_model_bert_reval.eval() # Colocar em modo de avaliação aqui é uma boa prática

    # --- Re-carregar e Re-processar Dados de Validação ESPECIFICAMENTE para esta seção de Reavaliação ---
    print("\n--- Re-carregando e Re-processando Dados de Validação para BERT (Reavaliação) ---")
    try:
        df_val_reval_load_bert = pd.read_csv('twitter_validation.csv', header=None, names=['TweetID', 'Entity', 'Sentiment', 'TweetContent'])
    except FileNotFoundError:
        print("Arquivo 'twitter_validation.csv' não encontrado para reavaliação do BERT. Faça o upload.")
        df_val_reval_load_bert = None

    if df_val_reval_load_bert is not None:
        df_val_reval_load_bert.dropna(subset=['TweetContent', 'Sentiment', 'Entity'], inplace=True)
        # Aplicar a limpeza de texto
        df_val_reval_load_bert['TweetContent_Cleaned'] = df_val_reval_load_bert['TweetContent'].apply(clean_text_for_bert)
        # Preparar listas de texto
        val_texts_entity_reval_load = df_val_reval_load_bert['Entity'].astype(str).tolist()
        val_texts_tweet_reval_load = df_val_reval_load_bert['TweetContent_Cleaned'].astype(str).tolist()
        # Codificar os sentimentos
        df_val_reval_load_bert['Sentiment_Encoded'] = df_val_reval_load_bert['Sentiment'].map(sentiment_map)
        df_val_reval_load_bert.dropna(subset=['Sentiment_Encoded'], inplace=True)
        val_labels_reval_load = df_val_reval_load_bert['Sentiment_Encoded'].values.astype(int)

        # Tokenizar dados de validação USANDO O TOKENIZER BERT CARREGADO
        val_input_ids_reval_load, val_attention_masks_reval_load, val_token_type_ids_reval_load = bert_encode(
            val_texts_entity_reval_load, val_texts_tweet_reval_load, loaded_tokenizer_bert_reval, MAX_LEN_BERT
        )
        val_labels_tensor_reval_load = torch.tensor(val_labels_reval_load)

        # Criar DataLoader de validação
        # Adicionar imports para TensorDataset, DataLoader, SequentialSampler se não estiverem no escopo
        from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
        val_dataset_reval_load = TensorDataset(val_input_ids_reval_load, val_attention_masks_reval_load, val_token_type_ids_reval_load, val_labels_tensor_reval_load)
        val_sampler_reval_load = SequentialSampler(val_dataset_reval_load)
        val_dataloader_reval_load = DataLoader(val_dataset_reval_load, sampler=val_sampler_reval_load, batch_size=BATCH_SIZE_BERT_EVAL)

        print(f"Dados de validação para reavaliação BERT processados. Número de batches: {len(val_dataloader_reval_load)}")

        # --- Reavaliação com Modelo Carregado ---
        print("\n--- Reavaliando o modelo BERT carregado ---")
        loaded_model_bert_reval.eval() # Certificar que está em modo de avaliação
        all_preds_bert_reval_final, all_true_labels_bert_reval_final = [], []
        total_eval_loss_bert_reval_final = 0
        total_eval_accuracy_bert_reval_final = 0

        for batch in val_dataloader_reval_load:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_token_type_ids = batch[2].to(device) # Se usou token_type_ids
            b_labels = batch[3].to(device)

            with torch.no_grad():
                outputs = loaded_model_bert_reval(b_input_ids,
                                           token_type_ids=b_token_type_ids, # Inclua se usou
                                           attention_mask=b_input_mask,
                                           labels=b_labels) # Fornecer labels calcula a loss
            loss = outputs.loss
            logits = outputs.logits

            total_eval_loss_bert_reval_final += loss.item()
            logits_detached = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            total_eval_accuracy_bert_reval_final += flat_accuracy(logits_detached, label_ids) # Usa flat_accuracy global
            all_preds_bert_reval_final.extend(np.argmax(logits_detached, axis=1).flatten())
            all_true_labels_bert_reval_final.extend(label_ids.flatten())

        avg_val_accuracy_reval = total_eval_accuracy_bert_reval_final / len(val_dataloader_reval_load)
        avg_val_loss_reval = total_eval_loss_bert_reval_final / len(val_dataloader_reval_load)

        print(f"  Acurácia na validação (BERT Reavaliação com Modelo Carregado): {avg_val_accuracy_reval:.4f}")
        print(f"  Perda (Loss) na validação (BERT Reavaliação com Modelo Carregado): {avg_val_loss_reval:.4f}")

        y_pred_classes_bert_reval_final = np.array(all_preds_bert_reval_final)
        y_val_bert_reval_final = np.array(all_true_labels_bert_reval_final)

        target_names_bert_reval = [sentiment_map_inv[i] for i in range(num_classes)] # Usa sentiment_map_inv e num_classes globais
        # Adicionar imports para classification_report e confusion_matrix se ainda não estiverem no escopo
        from sklearn.metrics import classification_report, confusion_matrix
        print("\nRelatório de Classificação (BERT Reavaliação com Modelo Carregado):")
        print(classification_report(y_val_bert_reval_final, y_pred_classes_bert_reval_final, target_names=target_names_bert_reval, zero_division=0))

        conf_matrix_bert_reval = confusion_matrix(y_val_bert_reval_final, y_pred_classes_bert_reval_final)
        # Adicionar imports para plt e sns se ainda não estiverem no escopo
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(8,6))
        sns.heatmap(conf_matrix_bert_reval, annot=True, fmt='d', cmap='Greens', xticklabels=target_names_bert_reval, yticklabels=target_names_bert_reval)
        plt.xlabel('Predito'); plt.ylabel('Verdadeiro'); plt.title('Matriz de Confusão - BERT (Reavaliação com Modelo Carregado)'); plt.show()
    else:
        print("Não foi possível carregar 'twitter_validation.csv' para a reavaliação do BERT.")
else:
    print(f"Diretório do modelo BERT salvo ('{MODEL_SAVE_DIR_BERT}') não encontrado para carregar.")
    print("Execute a seção de treinamento e salvamento do script completo primeiro, ou faça upload do diretório salvo se já o tiver.")

print("### FIM DA SEÇÃO DE CARREGAMENTO E REAVALIAÇÃO DO BERT ###")