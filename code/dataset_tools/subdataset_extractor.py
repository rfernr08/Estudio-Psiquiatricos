import polars as pd
dataset_path = "recursos/otros/diagnosticos_unidos.csv"
dataset = pd.read_csv(dataset_path, separator="|")

filtrado = dataset.filter(
    (dataset['DIAG PSQ'].str.contains('F20')) |
    (dataset['DIAG PSQ'].str.contains('F20.89'))
)

filtrado.write_csv('recursos/otros/diagnosticos_F20_F20.89.csv', separator="|")
