#!/usr/bin/bash 

#Remove all csv files 
echo "Removing all csv files"
rm *.csv

echo "Running script"
./build/clustering

python3 plot.py --path .
