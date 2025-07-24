import pandas as pd
import time
import numpy as np
import os
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import polars as pl
from sklearn.preprocessing import LabelEncoder
os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"

     
dataset =  "recursos/otros/diagnosticos_F20_F20.89.csv"
df = pl.read_csv(dataset, separator="|")

seed = int(time.time_ns() % (2**32))
np.random.seed(seed)
diag_colms = [col for col in df.columns if col.startswith("Diag")]

df_diag = df.select(diag_colms).fill_nan("").fill_null("")
X = df_diag.to_pandas()

# Encode categorical features
le = LabelEncoder()
for col in X.columns:
    X[col] = le.fit_transform(X[col].astype(str))

y = df.select(
    pl.col("DIAG PSQ").str.contains("F20.89").cast(pl.Int32).alias("target")
).to_pandas()["target"]

def create_text_from_diagnostics(row):
    diag_values = [str(val) for val in row if str(val) != '']
    return " ".join(diag_values)

# Apply to your encoded DataFrame
X['text'] = X.apply(create_text_from_diagnostics, axis=1)

# Create DataFrame with text and labels
df_bert = pd.DataFrame({'text': X['text'], 'label': y})

# Now split this DataFrame
X_train, X_test = train_test_split(df_bert, test_size=0.3, random_state=seed)

# Reset index por seguridad
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)


from datasets import Dataset
from transformers import AutoTokenizer

# Puedes usar modelos como:
# - 'bert-base-uncased'
# - 'dccuchile/bert-base-spanish-wwm-cased' (si es español)
# - 'PlanTL-GOB-ES/roberta-base-biomedical-clinical-es' (clínico en español)

model_name = 'PlanTL-GOB-ES/roberta-base-biomedical-clinical-es'
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Convertir a Hugging Face Dataset
train_dataset = Dataset.from_pandas(X_train)
test_dataset = Dataset.from_pandas(X_test)

# Tokenización
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

train_tokenized = train_dataset.map(tokenize_function, batched=True)
test_tokenized = test_dataset.map(tokenize_function, batched=True)

# Eliminar columnas no necesarias
train_tokenized = train_tokenized.remove_columns(["text"])
test_tokenized = test_tokenized.remove_columns(["text"])
train_tokenized.set_format("torch")
test_tokenized.set_format("torch")

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

from transformers import TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Métricas personalizadas
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='binary')
    acc = accuracy_score(p.label_ids, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Entrenar
trainer.train()

# Evaluación en test
metrics = trainer.evaluate()
print(metrics)

# Predicción
preds = trainer.predict(test_tokenized)
pred_labels = np.argmax(preds.predictions, axis=1)
