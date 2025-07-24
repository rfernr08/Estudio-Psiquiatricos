import pandas as pd
from transformers import AutoTokenizer
from EsquizofreniaDataset import EsquizofreniaDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from transformers import BertForSequenceClassification
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import torch
from sklearn.metrics import classification_report, confusion_matrix

# Cargar tu CSV o DataFrame
df = pd.read_csv("recursos/otros/diagnosticos_F20_F20.89_con_descripcion.csv", sep="|")

# Reemplaza espacios innecesarios en los nombres de columna
df.columns = df.columns.str.strip()

# Unir las 20 columnas de diagnóstico en una sola cadena de texto por fila
diag_columns = [col for col in df.columns if col.startswith("Diag")]
df["text"] = df[diag_columns].fillna("").agg(" ".join, axis=1)

print("Unique values in DIAG PSQ:", df["DIAG PSQ"].unique())

df["label"] = df["DIAG PSQ"].map({
    "Esquizofrenia": 1,
    "Otros tipos de esquizofrenia": 0
})

# Remove rows with NaN labels
df = df.dropna(subset=["label"])
print(f"Dataset size after removing NaN labels: {len(df)}")

# Convert to int to ensure no floating point issues
df["label"] = df["label"].astype(int)


tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
labels = df["label"].tolist()
# Tokenizamos los textos
encodings = tokenizer(
    df["text"].tolist(),
    truncation=True,
    padding=True,
    max_length=512,
    return_tensors="pt"
)



dataset = EsquizofreniaDataset(encodings, labels)



# Índices para separar
train_indices, test_indices = train_test_split(
    list(range(len(dataset))),
    test_size=0.2,
    stratify=labels,
    random_state=42
)

# Crear subsets
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

# Cargar en DataLoader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)


model = BertForSequenceClassification.from_pretrained(
    "dccuchile/bert-base-spanish-wwm-cased",
    num_labels=2  # 2 clases: F20 (1) y F20.89 (0)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizador
optimizer = AdamW(model.parameters(), lr=2e-5)

# Entrenamiento simple
model.train()
for epoch in range(3):  # puedes ajustar el número de épocas
    total_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    
    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1} loss: {total_loss / len(train_loader)}")

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch["labels"].cpu().numpy())

# Resultados
print(classification_report(all_labels, all_preds, target_names=["F20.89", "F20"]))
print(confusion_matrix(all_labels, all_preds))
