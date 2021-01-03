# Clustering with Iteratively Reweighted Descent

## Folder Structure  
The project source code is saved under __src__ folder. You can find all relevant _.cpp_ and _.h_ files there.  
Before running the algorithm you have to put the csv files for configuration in the folder __config__. Please use the given template to fill the configuration files there.  
If you want to use the data you have created, please put them in the __data__ folder and follow the instructions in Section __Run the Code__.  
The data that is presented in the report can be found in the folder __cases__.


## Run the Code 
- __Build Project:__  
  Use the following command lines to build the project:

  `mkdir build`  
  `cd build`  
  `cmake --configure ..`  
  `cmake --build <path-to-build-file> --target all`


- __Create Data:__
  Change the _variances.csv_ and _centers.csv_ files according to your choices and use the command line  
  `./build/clustering 0`  
  The data will be saved in the __data__ folder as _csv_ files.
- __Run Algorithm__
  To run the algorithm you need to specify the configuration parameters located in __config__ folder. After choosing the parameters you want to run the algorithm with, run the command 
  `./build/clustering 1`  
  This command will generate several _csv_ files containing the output of the algorithm.
- __Run Algorithm for Different Parameters:__
  You can also run the algorithm for several times with different _rZ_ parameters. Please specify the set of _rZ_ parameters in the __run_hyp.sh__ in the _paramRz_ array and how many times you want to run a configuration set repeatedly in **ARG_NR_EXPERIMENT**. To start the loop, use the command:  
  `./run_hyp.sh`  
