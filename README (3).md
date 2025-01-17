
# Sentiment Analysis with BERT

Ce projet utilise le modèle pré-entraîné **BERT** pour effectuer une analyse de sentiment sur des critiques de films provenant du **dataset IMDB**. L'objectif est de prédire si une critique est **positive** ou **négative** en utilisant BERT, un modèle de langage transformer très puissant.

## Prérequis

Avant de commencer, assurez-vous d'avoir installé les dépendances suivantes :

- `pandas` : Pour la manipulation des données.
- `re` : Pour le nettoyage des données textuelles (élimination de HTML, ponctuation, etc.).
- `torch` : Pour l'utilisation de PyTorch, la bibliothèque utilisée pour l'entraînement du modèle.
- `transformers` : Pour le modèle BERT et la tokenisation.
- `sklearn` : Pour la séparation des données en ensembles d'entraînement et de test.

### Installation des dépendances

```bash
pip install pandas torch transformers scikit-learn
```

## Description du projet

Le projet utilise le **dataset IMDB** pour analyser les critiques de films. Chaque critique est étiquetée comme étant **positive** ou **négative**. Nous allons entraîner un modèle BERT pour prédire le sentiment des critiques.

### Étapes principales :

1. **Prétraitement des données** :
   - Le dataset IMDB contient des critiques de films avec des balises HTML. Ces balises sont supprimées et le texte est converti en minuscules.
   - Les critiques sont ensuite tokenisées à l'aide du tokenizer BERT (`bert-base-uncased`).
   
2. **Tokenisation des critiques** :
   - Les critiques sont transformées en **input_ids** et en **attention_mask** pour être utilisées comme entrée du modèle BERT.
   
3. **Séparation des données** :
   - Le dataset est séparé en un ensemble d'entraînement (80%) et un ensemble de test (20%) à l'aide de la fonction `train_test_split` de `sklearn`.

4. **Création du Dataset pour HuggingFace** :
   - Un dataset personnalisé est créé pour être compatible avec les APIs de `HuggingFace`, en utilisant la classe `Dataset` de PyTorch.
   
5. **Entraînement du modèle** :
   - Le modèle `BertForSequenceClassification` est chargé et utilisé pour l'entraînement. Il est configuré pour avoir deux labels (`positive` et `negative`).
   - L'entraînement se fait en 3 époques, avec une taille de batch de 8.

6. **Évaluation du modèle** :
   - Le modèle est évalué sur l'ensemble de test et les résultats sont affichés.

7. **Sauvegarde du modèle** :
   - Une fois l'entraînement terminé, le modèle et le tokenizer sont sauvegardés dans un répertoire spécifique.

## Code d'entraînement

Le code Python pour entraîner le modèle est le suivant :

```python
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
```

## Résultats

Après l'entraînement, le modèle peut être utilisé pour prédire le sentiment d'une critique donnée. Les résultats de l'évaluation du modèle sont affichés dans la sortie du terminal.

## Conclusion

Ce projet montre comment utiliser un modèle pré-entraîné BERT pour effectuer une tâche d'analyse de sentiment. Le code inclut le prétraitement des données, la tokenisation, la séparation des données en ensembles d'entraînement et de test, ainsi que l'entraînement et l'évaluation du modèle BERT. 

Le modèle peut être utilisé pour classer de nouvelles critiques de films en fonction de leur sentiment (positif ou négatif).
