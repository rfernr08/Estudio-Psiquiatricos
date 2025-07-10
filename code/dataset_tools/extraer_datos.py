import csv

input_file = 'cie10-es-diagnoses.csv'      # Tu archivo original
output_file = 'ICD_10_Español.csv'         # Archivo de salida

with open(input_file, mode='r', newline='', encoding='utf-8') as infile, \
     open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
    reader = csv.DictReader(infile)
    writer = csv.DictWriter(outfile, fieldnames=['code', 'description'], delimiter='|')
    writer.writeheader()
    for row in reader:
        writer.writerow({'code': row['code'], 'description': row['description']})