#include <iostream>
#include <eigen3/Eigen/Core>
#include <chrono>
#include <random>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "Data.h"
#include "Trainer.h"
#include "Observation.h"

using namespace std;
using namespace Eigen;

unsigned random_seed = std::chrono::system_clock::now().time_since_epoch().count();    

//TODO: Read a Hashmap from CSV
//TODO: Add writeCSV for clean code

void readCSV(string path, double &r_x, double &r_z, double &clusterVariance1, double &clusterVariance2, double &clusterCenter1, double &clusterCenter2,double &beta){
    ifstream in(path);
    unsigned int counter = 0;
    if (in) {
        string line;
        unsigned int counter = 0;
        while (getline(in, line)) {
            stringstream sep(line);
            string field;
            string field_name;

            while (getline(sep, field, ',')) {

                switch(counter){
                    case 0:
                        r_x = stod(field);
                        break;
                    case 1:
                        r_z = stod(field);
                        break;
                    case 2: 
                        clusterVariance1 = stod(field);
                        break;
                    case 3: 
                        clusterVariance2 = stod(field);
                        break;
                    case 4:
                        clusterCenter1 = stod(field);
                        break;
                    case 5:
                        clusterCenter2 = stod(field);
                        break;
                    case 6:
                        beta = stod(field);
                        break;
                    
                    default:
                        break;        
                }

            }
            ++counter;

        }

    }
}

int main(){
    //Input parameters
    // string path = "/home/ander/Documents/git/clustering/inputConfig.txt";
    string path = "inputConfig.txt";

    double r_x, r_z, clusterVariance1, clusterVariance2, clusterCenter1, clusterCenter2, beta;
    readCSV(path, r_x, r_z, clusterVariance1, clusterVariance2, clusterCenter1, clusterCenter2, beta);
    bool COMPUTE_DATA_RZ = false;

    double learning_rate = 0.0001;
    unsigned int numberOfIterations = 1000;
    double tolerance = 0.00001;

    bool trainingMode = IRCD;
    unsigned int lossFunctionX = NUV;
    unsigned int lossFunctionZ = SNUV;

    unsigned int numberClusters = 2;
    unsigned int numberSamples = 100;



    
    



    // Define cluster centers and variances 
    VectorXd x(numberClusters);
    x << clusterCenter1, clusterCenter2;

    VectorXd variances(numberClusters);
    variances << clusterVariance1, clusterVariance2;






    Observation input_observation(x, variances, numberSamples, random_seed);

    Data example =  Data(input_observation, r_x, r_z, random_seed, beta);
    if(COMPUTE_DATA_RZ){
        double mean = 0.0;
        double variance = 0.0;
        double std = 0.0;
        for(int i=0; i<example.numberSamples;i++){
            mean += example.y(i);
        }
        mean /= example.numberSamples;

        for(int i=0; i<example.numberSamples;i++){
            variance += (example.y(i)*example.y(i)) - (mean*mean) ;
        }
        variance /= example.numberSamples;
        r_z= sqrt(variance);
        example.r_z = r_z;

    }
    
    //Create config.csv 
    ofstream configData;
    // configData.open("/home/ander/Documents/git/clustering/config.csv",ios::app);
    configData.open("config.csv",ios::app);
    configData << "learning_rate" << "," << learning_rate << endl;
    configData << "numberOfIterations" << "," << numberOfIterations << endl;
    configData << "tolerance" << "," << tolerance << endl;
    configData << "trainingMode" << "," << trainingMode << endl;
    configData << "lossFunctionX" << "," << lossFunctionX << endl;
    configData << "lossFunctionZ" << "," << lossFunctionZ << endl;
    configData << "numberClusters" << "," << numberClusters << endl;
    configData << "numberSamples" << "," << numberSamples << endl;
    configData << "r_x" << "," << r_x << endl;
    configData << "r_z" << "," << r_z << endl;
    configData << "clusterCenter1" << "," << clusterCenter1 << endl;
    configData << "clusterVariance1" << "," << clusterVariance1 << endl;
    configData << "clusterCenter2" << "," << clusterCenter2 << endl;
    configData << "clusterVariance2" << "," << clusterVariance2 << endl;
    configData << "beta" << "," << beta << endl;
    configData.close();

    Trainer trainer(trainingMode, lossFunctionX, lossFunctionZ,learning_rate, numberOfIterations, tolerance);
    trainer.train(example);
    cout << " R_Z " << r_z << endl;
    

    return 0;

}