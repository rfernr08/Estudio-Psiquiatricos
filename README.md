# Estudio-Psiquiatricos

Procesamiento y analisis de un dataset sobre diagnosticos psiquiatricos

# Codigos Usados:

# Csv manger

-code/csv_manager/borrar_dups_csv.py
Borra las lineas duplicadas de un csv, ya que la lista de relaciones de diagnosticos tenia algunas lineas que se repetian al aparecer varias veces la misma relacion en el dataset

-code/csv_manager/excel_a_csv.py
Codigo para poder pasar el excel del dataset a csv para un manejo mas comodo

-code/csv_manager/juntar_csv.py
Codigo que montaba las descripciones que tenia un csv, a las columnas de un csv con amyor numero de codigos

-code/csv_manager/mergue_rapido.py
Codigo sencillo para juntar 2 csv identicos en uno mayor

-code/csv_manager/merge_csv.py
Codigo que primero limpia los puntos sobrantes de un csv que tenia buena parte de los codigos ICD-10 y ICD-9 que existen
Deespues junta con un join 2 csv de codigos y forma una lista larga de codigos de diagnosticos con sus 2 variantes, algunos con descripcion.

# Dataset_tools

-code/dataset_tools/conversor-ICD.py
Codigo que recorre todo un dataset. leyendo las columnas de diagnsoticos indivuales, diagnsotico psiquiatrico y conjunto de diagnsoticos, comprobando que esten
todos los codigos en ICD-10 y si no pasandolos a ICD-10 (en caso de que esten en ICD-9)
Finalemente añade nuevas columnas con las descripciones en español, ingles y variante en ICD-9 de cada codigo encontrado en el dataset.

-code/dataset_tools/extraer_diag.py
Extrae las columnas de diagnsoticos individuales y diagnsticos psquiatricos para poder sacar una lista con todos los diagnosticos encontrados en el 
dataset y las relaciones entre ellos.

-code/dataset_tools/relacionador.py
Codiogo que contruira la lista de diagnsoticos que se encuentran con diagnsoticos psquiatricos

-code/dataset_tools/lista_diag_multi.py
Codigo para construir la informacion de cada diagnsoticos encontrado en el dataset.
Descripcion en español|Codigo en ICD-10|Codigo en ICD-9|Descripcion en ingles|Embedding
El embedding lo genera el metodo mismamente usando un all-MiniLM-L6-v2 de sentence transformer
