#!/bin/bash

folder="/root/lanyun-fs/team2/team2_data/token"
files=($(ls "$folder"))
num_files=${#files[@]}

for ((j=0; i<1; j++)); do
  for ((i=0; i<$num_files; i++)); do
    fn="${files[$i]}"

    if [ "$i" -gt 170 ]; then
      deepspeed pretrain.py --data_file "$folder/$fn" --ss $i
      echo "file $fn is done..."
    fi
  done
done
