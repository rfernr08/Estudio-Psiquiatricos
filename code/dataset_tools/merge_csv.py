

import polars as pl
import re

def limpiar_codigos_con_punto_final(path_csv_entrada: str, path_csv_salida: str) -> None:
    """
    Limpia un CSV separador por '|', eliminando los puntos finales de códigos ICD9 e ICD10
    cuando no van seguidos de un número (ej: 'A94.' -> 'A94').
    
    Guarda un nuevo CSV limpio en path_csv_salida.
    """
    df = pl.read_csv(
        path_csv_entrada,
        separator='|',
        has_header=False,
        new_columns=["ICD9", "ICD10"],
        schema_overrides={"ICD9": pl.Utf8, "ICD10": pl.Utf8}
    )

    def limpiar_codigo(codigo: str) -> str:
        if isinstance(codigo, str) and re.match(r".+\.\s*$", codigo):
            return codigo.rstrip(". ")
        return codigo

    df_limpio = df.with_columns([
        pl.col("ICD9").map_elements(limpiar_codigo).alias("ICD9"),
        pl.col("ICD10").map_elements(limpiar_codigo).alias("ICD10"),
    ])

    df_limpio.write_csv(path_csv_salida, separator='|')


def fusionar_csv_backup_formated(path_backup: str, path_formated: str, salida: str) -> None:
    """
    Fusiona el CSV 'formated' (con descripciones) con 'backup' (sin descripciones),
    conservando todos los códigos del backup y añadiendo descripciones si existen.
    """
    df_backup = pl.read_csv(
        path_backup,
        separator="|",
        has_header=False,
        new_columns=["ICD9", "ICD10"],
        schema_overrides={"ICD9": pl.Utf8, "ICD10": pl.Utf8}
    )

    df_formated = pl.read_csv(
        path_formated,
        separator="|",
        has_header=False,
        new_columns=["ICD9", "ICD10", "Description"],
        schema_overrides={"ICD9": pl.Utf8, "ICD10": pl.Utf8, "Description": pl.Utf8}
    )

    # Fusionar: mantener todo el backup, añadir descripciones si existen
    df_merged = df_backup.join(df_formated, on=["ICD9", "ICD10"], how="left")

    df_merged = df_merged.with_columns(
        pl.col("Description").fill_null("Diagnostico no disponible")
    )

    # Guardar el resultado
    df_merged.write_csv(salida, separator="|")

limpiar_codigos_con_punto_final("backup_conversor.csv", "backup_limpio.csv")

fusionar_csv_backup_formated(
    path_backup="backup_limpio.csv",
    path_formated="ICD10_Formatted.csv",
    salida="ICD10_Completo.csv"
)
