#!/usr/bin/bash 
ARG_START=1
ARG_K=$1

ARG_DIR=$(pwd)'/'
if [ $(ls experiments | wc -l) -eq 0 ]; then
    NUMBER=0
    echo "if"
    
else 
    #Get the last NUMBER of experiment file 
    declare -a arr=( )
    ARG_PATH=$ARG_DIR'experiments'
    echo $ARG_PATH
    paths=( $(ls $ARG_PATH) )
    for index in ${!paths[*]}
    do
        tmp=$(echo "${paths[$index]}" | sed 's/[^0-9]*//g')
        arr+=($tmp)

    done
    #Find the largest suffix
    IFS=$'\n'
    NUMBER=$(echo "${arr[*]}" | sort -nr | head -n1)
    NUMBER=$(($NUMBER + 1))
fi

mkdir experiments
cp -r config experiments
cp -r data experiments

EXPERIMENT='experiments/experiment'
ARG_RESULT="$EXPERIMENT""$NUMBER"

for ((i = 1; i <= $ARG_K; i++))
do
    echo "RUN $i IS STARTING"

    mkdir -p "$ARG_RESULT/run$i"

    # #Remove all csv files 
    echo "Removing all csv files"
    rm *.csv

    # echo "Running script"
    # if data exists use 1 
    ./build/clustering 1
    # to create data run ./build/clustering 0

    python3 plot_experiment.py --path .


    # Find the largest suffix in experiments folder and create a new folder with largest suffix + 1


    echo "Moving files to $ARG_RESULT"
    mv *.csv *.png -t $ARG_RESULT"/run$i"
    echo "RUN $i IS DONE"

done 