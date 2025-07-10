import pandas as pd

df = pd.read_csv("datasets/importante/Full_Datos_Diag_Men.csv", sep="|")

df.to_csv("datasets/importante/Full_Datos_Diag_Men.csv", sep="|", index=False)