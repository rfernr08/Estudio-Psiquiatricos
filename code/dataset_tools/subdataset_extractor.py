import polars as pl
dataset_path = "recursos/otros/BERT/diagnosticos_unidos.csv"
dataset = pl.read_csv(dataset_path, separator="|")

filtrado = dataset.filter(
    (dataset['DIAG PSQ'].str.contains('F20')) |
    (dataset['DIAG PSQ'].str.contains('F20.89'))
)

for col in filtrado.columns:
    if col != 'DIAG PSQ' and filtrado[col].dtype == pl.String:
        filtrado = filtrado.with_columns(
            pl.when(filtrado[col].str.contains('F20.89'))
            .then(pl.lit(None))
            .when(filtrado[col].str.contains('F20.0'))
            .then(pl.lit(None))
            .otherwise(filtrado[col])
            .alias(col)
        )

filtrado.write_csv('recursos/otros/BERT/diagnosticos_F20_F20.89_sin_dups.csv', separator="|")
