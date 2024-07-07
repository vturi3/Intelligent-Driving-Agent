#!/bin/bash

# Numero di iterazioni desiderate
iterazioni=10

# Percorso dello script da lanciare
script_path="./run_test.sh"

# Percorso della cartella dei risultati
results_folder="./results"

# Nome del file di risultato
result_file="simulation_results.json"

# Esegui le iterazioni
for ((i=1; i<=iterazioni; i++)); do
 # Esegui lo script
 bash "$script_path"

 # Rinomina il file di risultato includendo il numero di iterazione
 new_file_name="result_${i}.json"
 mv "${results_folder}/${result_file}" "${results_folder}/${new_file_name}"
done



echo "Simulazioni completate!"