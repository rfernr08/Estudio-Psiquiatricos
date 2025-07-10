import csv

input_csv = 'datasets/Conversor_Definitivo.csv'
output_csv = 'diagnosticos_icd9_letras.csv'

with open(input_csv, newline='', encoding='utf-8') as infile, \
    open(output_csv, 'w', newline='', encoding='utf-8') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    header = next(reader)
    writer.writerow(header)
    for row in reader:
       codigo = row[0].strip()
       print(f"Procesando codigo: {codigo}")
       if codigo and codigo[0].isalpha() and codigo[0] != 'N':
          writer.writerow(row)