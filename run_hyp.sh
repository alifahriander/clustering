#!/usr/bin/bash 
# paramsBeta=(0.01 0.1 1.0 10.0 100.0)
# paramsRx=(0.01 0.1 1.0 10.0 100.0)

paramsBeta=(0.01)
paramsRx=(0.01)
paramsRz=(0.01 0.1 1.0 10.0 100.0 1000.0 10000.0)

# paramsRz=(0.01)
# paramsRx=(0.01)
# paramsBeta=(0.0001 0.001 0.01 0.1 1.0 10.0 100.0 1000.0 10000.0)

LINE_BETA=7
LINE_RX=1
LINE_RZ=2

ARG_NR_EXPERIMENTS=10

CONFIG_FILE="inputConfig.txt"
TMP_FILE="conf.txt"

for beta in ${paramsBeta[@]}; do
    for rx in ${paramsRx[@]}; do
        for rz in ${paramsRz[@]}; do
            # Change parameters
            awk -v varRx="$rx" -v varLineRx="$LINE_RX" 'NR==varLineRx {$0=varRx} { print }' $CONFIG_FILE > $TMP_FILE 
            mv $TMP_FILE $CONFIG_FILE 
            awk -v varRz="$rz" -v varLineRz="$LINE_RZ" 'NR==varLineRz {$0=varRz} { print }' $CONFIG_FILE > $TMP_FILE
            mv $TMP_FILE $CONFIG_FILE 
            awk -v varBeta="$beta" -v varLineBeta="$LINE_BETA" 'NR==varLineBeta {$0=varBeta} { print }' $CONFIG_FILE > $TMP_FILE
            mv $TMP_FILE $CONFIG_FILE 


            # echo "Hyperparameter Combination(rx,rz,beta):"
            echo "=========" 

            echo $rx $rz $beta 
            cat $CONFIG_FILE

            ./run.sh $ARG_NR_EXPERIMENTS
        done
    done  
done