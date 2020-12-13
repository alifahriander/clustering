#!/usr/bin/bash 
paramsRz=(0.000001 0.00001 0.0001 0.001 0.01 0.1 1.0 10.0 100.0 1000.0 10000.0)



ARG_NR_EXPERIMENTS=3
CONFIG_FILE="./config/r_z.csv"



for rz in ${paramsRz[@]}; do
    # Change parameters
    echo "$rz" > $CONFIG_FILE
    cat $CONFIG_FILE
    # echo "Hyperparameter Combination(rx,rz,beta):"
    echo "=========" 
    ./run.sh $ARG_NR_EXPERIMENTS
done

