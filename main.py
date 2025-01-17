import pandas as pd
import re
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

# Initialisation du tokenizer BERT
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Chargement du fichier CSV
df = pd.read_csv("IMDB Dataset.csv")
df = pd.DataFrame(df)

# Prétraitement des critiques (enlever HTML, ponctuation, mettre en minuscules)
df['review'] = df['review'].apply(lambda x: re.sub(r'<[^>]+>', '', x))
df['review'] = df['review'].apply(lambda x: re.sub(r'[^\w\s]', '', x).lower())

# Tokenisation des critiques
df['input_ids'] = df['review'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True, truncation=True, padding='max_length', max_length=512))
df['attention_mask'] = df['input_ids'].apply(lambda x: [1 if i != 0 else 0 for i in x])
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Affichage des 5 premières lignes
print(df[['review', 'input_ids', 'attention_mask', 'label']].head(5))

# Séparation des caractéristiques et des labels
X = df[['input_ids', 'attention_mask']]
y = df['label']

# Split des données en train et test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Ensemble d'entraînement: {len(X_train)} critiques")
print(f"Ensemble de test: {len(X_test)} critiques")

# Conversion des données en tensors
X_train_input_ids = torch.tensor(X_train['input_ids'].to_list())
X_train_attention_mask = torch.tensor(X_train['attention_mask'].to_list())
y_train = torch.tensor(y_train.to_list())

X_test_input_ids = torch.tensor(X_test['input_ids'].to_list())
X_test_attention_mask = torch.tensor(X_test['attention_mask'].to_list())
y_test = torch.tensor(y_test.to_list())

# Création de la classe Dataset pour HuggingFace
class SentimentDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

# Création des datasets d'entraînement et de test
train_dataset = SentimentDataset(X_train_input_ids, X_train_attention_mask, y_train)
test_dataset = SentimentDataset(X_test_input_ids, X_test_attention_mask, y_test)

# Initialisation du modèle BERT pour la classification de séquences
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Définition des arguments d'entraînement
training_args = TrainingArguments(
    output_dir='./results',          # Où sauvegarder les résultats du modèle
    num_train_epochs=3,              # Nombre d'époques d'entraînement
    per_device_train_batch_size=8,   # Taille du batch pour l'entraînement
    per_device_eval_batch_size=8,    # Taille du batch pour l'évaluation
    warmup_steps=500,                # Nombre de steps pour le warmup
    weight_decay=0.01,               # L2 regularization
    logging_dir='./logs',            # Où sauvegarder les logs
    logging_steps=10,                # Fréquence d'enregistrement des logs
    evaluation_strategy="epoch",     # Évaluation chaque époque
    save_strategy="epoch",           # Sauvegarder chaque époque
)

# Initialisation du Trainer
trainer = Trainer(
    model=model,                       # Le modèle que nous avons chargé
    args=training_args,                # Les arguments d'entraînement
    train_dataset=train_dataset,       # Dataset d'entraînement
    eval_dataset=test_dataset          # Dataset de test
)

# Entraînement du modèle
trainer.train()

# Évaluation du modèle
results = trainer.evaluate(test_dataset)
print("Résultats de l'évaluation :", results)

# Sauvegarde du modèle et du tokenizer
model.save_pretrained('./my_bert_model')
tokenizer.save_pretrained('./my_bert_model')
