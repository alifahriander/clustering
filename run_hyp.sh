#!/usr/bin/bash 
paramsRz=(0.01 0.1 1.0 10.0 100.0 1000.0 10000.0)



ARG_NR_EXPERIMENTS=1
CONFIG_FILE="./config/r_z.csv"



for rz in ${paramsRz[@]}; do
    # Change parameters
    echo "$rz" > $CONFIG_FILE
    cat $CONFIG_FILE
    # echo "Hyperparameter Combination(rx,rz,beta):"
    echo "=========" 
    ./run.sh $ARG_NR_EXPERIMENTS
done

