#Obs: esse projeto fica melhor de visualizar e executar pelo colab, link: https://colab.research.google.com/drive/1zSn7fIKh8H7aZt08605EGcJmusZ-M0c3?usp=sharing


#!pip install transformers
#!pip install accelerate -U

import os
import pandas as pd
import numpy as np
import torch
import re

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset, DataLoader

torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

from transformers import AutoTokenizer, Trainer, TrainingArguments, BertForSequenceClassification, AdamW

#Treinamento para o Classificador - AMP e NÃO-AMP

class amp_data():
    def __init__(self, df, tokenizer_name='Rostlab/prot_bert_bfd', max_len=200):

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, do_lower_case=False)
        self.max_len = max_len

        self.seqs, self.labels = self.get_seqs_labels()

    def get_seqs_labels(self):
        seqs = list(df['aa_seq'])
        labels = list(df['AMP'].astype(int))

        return seqs, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = " ".join("".join(self.seqs[idx].split()))
        seq_ids = self.tokenizer(seq, truncation=True, padding='max_length', max_length=self.max_len)

        sample = {key: torch.tensor(val) for key, val in seq_ids.items()}
        sample['labels'] = torch.tensor(self.labels[idx])

        return sample
    
data_url = 'https://raw.githubusercontent.com/GIST-CSBL/AMP-BERT/main/all_veltri.csv'
df = pd.read_csv(data_url, index_col = 0)
df = df.sample(frac=1, random_state = 0)
print(df.head(7))

train_dataset = amp_data(df)



def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }
    
# define the model initializing function for Trainer in huggingface

def model_init():
    return BertForSequenceClassification.from_pretrained('Rostlab/prot_bert_bfd')

# training on entire data
# no evaluation/validation

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    learning_rate = 5e-5,
    per_device_train_batch_size=1,
    warmup_steps=0,
    weight_decay=0.1,
    logging_dir='./logs',
    logging_steps=100,
    do_train=True,
    do_eval=True,
    evaluation_strategy="no",
    save_strategy='no',
    gradient_accumulation_steps=64,
    fp16=True,
    fp16_opt_level="O2",
    run_name="AMP-BERT",
    seed=0,
    load_best_model_at_end = True
)

trainer = Trainer(
    model_init = model_init,
    args = training_args,
    train_dataset = train_dataset,
    compute_metrics = compute_metrics,
)

trainer.train()

# performance metrics on the training data itself

predictions, label_ids, metrics = trainer.predict(train_dataset)
metrics

# save the model, if desired

from google.colab import drive
drive.mount('/content/drive')

trainer.save_model('/content/drive/MyDrive/Colab Notebooks/AMP-BERT/Fine-tuned_model_MASK/')


#Load model from Drive

from google.colab import drive
drive.mount('/content/drive')

# load appropriate tokenizer and fine-tuned model
tokenizer = AutoTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False)
model = BertForSequenceClassification.from_pretrained("/content/drive/MyDrive/Colab Notebooks/AMP-BERT/Fine-tuned_model_2")

# predict AMP/non-AMP for a single example (default ex. is from external test data: DRAMP00126)

#@markdown **Input peptide sequence (upper case only)**
input_seq = 'YSCDFADSLSK' #@param {type:"string"}
input_seq_spaced = ' '.join([ input_seq[i:i+1] for i in range(0, len(input_seq), 1) ])
input_seq_spaced = re.sub(r'[UZOB]', 'X', input_seq_spaced)
input_seq_tok = tokenizer(input_seq_spaced, return_tensors = 'pt')

output = model(**input_seq_tok)
logits = output[0]

# extract AMP class probability and make binary prediction
y_prob = torch.sigmoid(logits)[:,1].detach().numpy()
y_pred = y_prob > 0.5
if y_pred == True:
  input_class = 'AMP'
else:
  input_class = 'non-AMP'

print('Input peptide sequence: ' + input_seq)
print('Class prediction: ' + input_class)


#Fine Tuning para o Método Mask do BERT - BertForMaskedLM
#CONFIGURAÇÃO E TREINAMENTO PARA FINE-TUNING

from transformers import BertTokenizer

class amp_data():
    def __init__(self, df, tokenizer_name='Rostlab/prot_bert_bfd', max_len=200):

        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name, do_lower_case=False)
        self.max_len = max_len

        self.seqs, self.labels = self.get_seqs_labels()

    def get_seqs_labels(self):
        seqs = list(df['aa_seq'])
        labels = list(df['AMP'].astype(int))

        return seqs, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = " ".join("".join(self.seqs[idx].split()))
        seq_ids = self.tokenizer(seq, truncation=True, padding='max_length', max_length=self.max_len)

        sample = {key: torch.tensor(val) for key, val in seq_ids.items()}
        sample['labels'] = torch.tensor(self.labels[idx])

        return sample
    
    
data_url = 'https://raw.githubusercontent.com/GIST-CSBL/AMP-BERT/main/all_veltri.csv'
df = pd.read_csv(data_url, index_col = 0)
df = df.sample(frac=1, random_state = 0)

new_df = df[df['AMP'] == True]

fine_tuning_dataset = amp_data(new_df)
new_df


from transformers import BertForMaskedLM, BertTokenizer, pipeline, AutoModelForMaskedLM, DataCollatorForLanguageModeling

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

tokenizer_fine = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer_fine, mlm_probability=0.15)


# define the model initializing function for Trainer in huggingface

def model_init():
    return AutoModelForMaskedLM.from_pretrained('Rostlab/prot_bert_bfd')


# TREINAMENTO PARA O DATASET DE FINE TUNING - APENAS PARA OS CASOS POSITIVOS DE AMP

fine_tuning_args = TrainingArguments(
    output_dir='./results_fine_tuning_amp',
    overwrite_output_dir=True,
    evaluation_strategy="no",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    fp16=True,
    logging_steps=100,
)

# fine_tuning_args = TrainingArguments(
#     output_dir='./results_fine_tuning_amp',
#     num_train_epochs=15,
#     learning_rate=5e-5,
#     per_device_train_batch_size=1,
#     warmup_steps=0,
#     weight_decay=0.1,
#     logging_dir='./logs_fine_tuning_amp',
#     logging_steps=100,
#     do_train=True,
#     do_eval=True,
#     evaluation_strategy="no",
#     save_strategy='no',
#     gradient_accumulation_steps=1,
#     fp16=True,
#     fp16_opt_level="O2",
#     run_name="AMP-BERT-fine-tuning",
#     seed=0,
#     load_best_model_at_end=True
# )


trainer_fine_tuning = Trainer(
    train_dataset=fine_tuning_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    args=fine_tuning_args,
    model_init= model_init,
)


trainer_fine_tuning.train()
# save the model, if desired
from google.colab import drive
drive.mount('/content/drive')

trainer_fine_tuning.save_model('/content/drive/MyDrive/AMP-BERT/Fine-tuned_model-MASK3/')
#**UTILIZAÇÃO DO MODELO COM FINE-TUNING REALIZADO**


# IMPORTAÇÃO DO MODELO DE MASK COM FINE TUNING

from transformers import BertTokenizer, BertForMaskedLM
from google.colab import drive
drive.mount('/content/drive')

tokenizer_fine = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
#tokenizer_fine = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False)
model_fine = BertForMaskedLM.from_pretrained("/content/drive/MyDrive/AMP-BERT/Fine-tuned_model-AMP_3")

# IMPORTAÇÃO DO MODELO DE CLASSIFICAÇÃO

from google.colab import drive
drive.mount('/content/drive')

tokenizer = AutoTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False)
model = BertForSequenceClassification.from_pretrained("/content/drive/MyDrive/Colab Notebooks/AMP-BERT/Fine-tuned_model")

from transformers import BertForMaskedLM, BertTokenizer, pipeline
import random


def substituir_caractere_por_mascara(input_seq):
    seq_com_mascara = list(input_seq)
    posicao_substituir = random.randint(0, len(seq_com_mascara) - 1)
    seq_com_mascara[posicao_substituir] = "#"
    return ''.join(seq_com_mascara)

def tratar_mask(input_seq):
    espaced = any(input_seq[i].isspace() for i in range(len(input_seq) - 1))
    if(not espaced):
        input_seq = input_seq.replace("[MASK]", "#")
        input_seq = ' '.join([ input_seq[i:i+1] for i in range(0, len(input_seq), 1) ])

    input_seq = input_seq.replace("#", "[MASK]")

    return input_seq


def classificar_sequencia_AMP(input_seq):
    input_seq_spaced = input_seq
    input_seq_tok = tokenizer(input_seq_spaced, return_tensors = 'pt')

    output = model(**input_seq_tok)
    logits = output[0]

    y_prob = torch.sigmoid(logits)[:,1].detach().numpy()
    y_pred = y_prob > 0.5
    if y_pred == True:
      input_class = 'AMP'
    else:
      input_class = 'non-AMP'

    return input_class, y_prob




unmasker = pipeline('fill-mask', model=model_fine, tokenizer=tokenizer_fine)

input_seq = 'SLFSLIK' #@param {type:"string"}

resultados = []

quantidade_resultados = 5

for _ in range(quantidade_resultados):
    sequencia_substituida = substituir_caractere_por_mascara(input_seq)
    sequencia_tratata = tratar_mask(sequencia_substituida)
    resultado = unmasker(sequencia_tratata)
    for sequencia in resultado:
      # Obter a sequência a ser classificada
      sequencia_input = sequencia['sequence']

      # Classificar a sequência
      classe_predita, probabilidade = classificar_sequencia_AMP(sequencia_input)

      # Adicionar a classificação ao objeto do resultado
      sequencia['AMP'] = classe_predita
      sequencia['probabilidade_AMP'] = probabilidade
    resultados.append(resultado)

print(resultados)

from transformers import BertForMaskedLM, BertTokenizer, pipeline
import random


def substituir_caractere_por_mascara(input_seq):
    seq_com_mascara = list(input_seq)
    posicao_substituir = random.randint(0, len(seq_com_mascara) - 1)
    seq_com_mascara[posicao_substituir] = "#"
    return ''.join(seq_com_mascara)

def tratar_mask(input_seq):
    espaced = any(input_seq[i].isspace() for i in range(len(input_seq) - 1))
    if(not espaced):
        input_seq = input_seq.replace("[MASK]", "#")
        input_seq = ' '.join([ input_seq[i:i+1] for i in range(0, len(input_seq), 1) ])

    input_seq = input_seq.replace("#", "[MASK]")

    return input_seq


def classificar_sequencia_AMP(input_seq):
    input_seq_spaced = input_seq
    input_seq_tok = tokenizer(input_seq_spaced, return_tensors = 'pt')

    output = model(**input_seq_tok)
    logits = output[0]

    y_prob = torch.sigmoid(logits)[:,1].detach().numpy()
    y_pred = y_prob > 0.5
    if y_pred == True:
      input_class = 'AMP'
    else:
      input_class = 'non-AMP'

    return input_class, y_prob




unmasker = pipeline('fill-mask', model=model_fine, tokenizer=tokenizer_fine)

input_seq = 'SLFSLIK' #@param {type:"string"}

resultados = []

quantidade_resultados = 5

for _ in range(quantidade_resultados):
    sequencia_substituida = substituir_caractere_por_mascara(input_seq)
    sequencia_tratata = tratar_mask(sequencia_substituida)
    resultado = unmasker(sequencia_tratata)
    for sequencia in resultado:
      # Obter a sequência a ser classificada
      sequencia_input = sequencia['sequence']

      # Classificar a sequência
      classe_predita, probabilidade = classificar_sequencia_AMP(sequencia_input)

      # Adicionar a classificação ao objeto do resultado
      sequencia['AMP'] = classe_predita
      sequencia['probabilidade_AMP'] = probabilidade
    resultados.append(resultado)

print(resultados)


import pandas as pd

dados = resultados

dataframes_por_array = {}

for idx, lista_sequencias in enumerate(dados):
    scores = []
    tokens = []
    token_strs = []
    sequences = []
    AMP = []

    for sequencia in lista_sequencias:
        scores.append(sequencia['score'])
        tokens.append(sequencia['token'])
        token_strs.append(sequencia['token_str'])
        sequences.append(sequencia['sequence'])
        AMP.append(sequencia['AMP'] == "AMP")

    data = {
        'score': scores,
        'token': tokens,
        'token_str': token_strs,
        'sequence': sequences,
        'AMP': AMP
    }

    dataframes_por_array[idx + 1] = pd.DataFrame(data)

for array_num, df in dataframes_por_array.items():
    display(df)
    
dados = resultados

def calcular_medias_assertividade(dados):
    num_sequencias = len(dados)
    num_posicoes = 3
    print(num_posicoes)

    medias_assertividade_por_posicao = [0] * num_posicoes

    for posicao in range(num_posicoes):
        soma_scores_posicao = 0
        for sequencia in dados:
            soma_scores_posicao += sequencia[posicao]['score']

        media_assertividade_posicao = soma_scores_posicao / num_sequencias
        medias_assertividade_por_posicao[posicao] = media_assertividade_posicao

    return medias_assertividade_por_posicao

medias_assertividade_por_posicao = calcular_medias_assertividade(dados)
for posicao, media in enumerate(medias_assertividade_por_posicao):
    print(f"Média de assertividade para a posição {posicao} nas sequências: {media * 100:.2f}%")
    
    
import matplotlib.pyplot as plt

dados = resultados


def plotar_grafico_assertividade(medias_assertividade):
    print("\n\n")
    num_posicoes = len(medias_assertividade)

    labels = [f"Posição {posicao}" for posicao in range(num_posicoes)]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, medias_assertividade, color='skyblue')

    plt.title('Médias de Assertividade por posição no resultado')

    plt.xlabel('Posição')
    plt.ylabel('Média de Assertividade')

    for index, value in enumerate(medias_assertividade):
        plt.text(index, value, f'{value:.2f}', ha='center', va='bottom')

    plt.show()

plotar_grafico_assertividade(medias_assertividade_por_posicao)

###


def plotar_grafico_resultados(dados):
    print("\n\n")
    num_top_resultados = 3
    num_sequencias = len(dados)
    tokens_str = [dados[0][i]['token_str'] for i in range(len(dados[0]))]

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, sequencia in enumerate(dados):
        top_resultados = sorted(sequencia, key=lambda x: x['score'], reverse=True)[:num_top_resultados]
        scores = [resultado['score'] for resultado in top_resultados]
        posicoes_tokens = [pos + i * (num_top_resultados + 1) for pos in range(len(top_resultados))]
        ax.bar(posicoes_tokens, scores, label=f'Sequência {i+1}')

    ax.set_xticks([])
    ax.set_xlabel('Tokens')
    ax.set_ylabel('Pontuação')
    ax.set_title('Top 3 Resultados para Cada Token nas Sequências')
    ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

plotar_grafico_resultados(dados)


###

def plotar_grafico_dispersao(dados):
    print("\n\n")
    num_top_resultados = 3

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, sequencia in enumerate(dados):
        top_resultados = sorted(sequencia, key=lambda x: x['score'], reverse=True)[:num_top_resultados]
        scores = [resultado['score'] for resultado in top_resultados]
        tokens = [resultado['token'] for resultado in top_resultados]
        ax.scatter(tokens, scores, label=f'Sequência {i+1}')

    ax.set_xticks([])
    ax.set_xlabel('Token')
    ax.set_ylabel('Pontuação')
    ax.set_title('Gráfico de Dispersão dos 3 Melhores Resultados para Cada Token nas Sequências')
    ax.legend()

    plt.tight_layout()
    plt.show()

plotar_grafico_dispersao(dados)


#Testes

#Original
from transformers import BertForMaskedLM, BertTokenizer, pipeline
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
model = BertForMaskedLM.from_pretrained("Rostlab/prot_bert")
unmasker = pipeline('fill-mask', model=model, tokenizer=tokenizer)
unmasker(input_seq)

#Classificador

#CLASSIFICADOR

input_seq = 'S L F S L I K A G A K F L G K N L L K Q G A K Y A A C K A S K Q L' #@param {type:"string"}
input_seq_spaced = input_seq
input_seq_tok = tokenizer(input_seq_spaced, return_tensors = 'pt')

output = model(**input_seq_tok)
logits = output[0]

# extract AMP class probability and make binary prediction
y_prob = torch.sigmoid(logits)[:,1].detach().numpy()
y_pred = y_prob > 0.5
print(y_pred, y_prob)
if y_pred == True:
  input_class = 'AMP'
else:
  input_class = 'non-AMP'

print('Input peptide sequence: ' + input_seq)
print('Class prediction: ' + input_class)



def classificar_sequencia_AMP(input_seq):
    input_seq_spaced = input_seq
    input_seq_tok = tokenizer(input_seq_spaced, return_tensors = 'pt')

    output = model(**input_seq_tok)
    logits = output[0]

    # extract AMP class probability and make binary prediction
    y_prob = torch.sigmoid(logits)[:,1].detach().numpy()
    y_pred = y_prob > 0.5
    if y_pred == True:
      input_class = 'AMP'
    else:
      input_class = 'non-AMP'

    return input_class, y_prob


input_seq = 'S L F S L I K A G A K F L G K N L L K Q G A K Y A A C K A S K Q L' #@param {type:"string"}

print(classificar_sequencia_AMP(input_seq))




from transformers import BertForMaskedLM, BertTokenizer, pipeline

unmasker = pipeline('fill-mask', model=model_fine, tokenizer=tokenizer_fine)
input_seq = 'SLFSLIKAGAKFLGKNLLKQGACYAACKASKQL' #@param {type:"string"}

espaced = any(input_seq[i].isspace() for i in range(len(input_seq) - 1))
if(not espaced):
    input_seq = input_seq.replace("[MASK]", "#")
    input_seq = ' '.join([ input_seq[i:i+1] for i in range(0, len(input_seq), 1) ])

input_seq = input_seq.replace("#", "[MASK]")

unmasker(input_seq)