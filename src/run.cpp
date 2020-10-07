#include <iostream>
#include <eigen3/Eigen/Core>
#include  <chrono>
#include  <random>
#include "Data.h"
#include "Trainer.h"
#include "Observation.h"






using namespace std;
using namespace Eigen;

unsigned random_seed = std::chrono::system_clock::now().time_since_epoch().count();    


int main(){
    
    unsigned int numberClusters = 2;
    unsigned int numberSamples = 100;

    // Define cluster centers and variances 
    VectorXd x(numberClusters);
    x << 10.0, 100.0;
    VectorXd variances(numberClusters);
    variances << 1.0, 1.0;

    cout << "Input x: "<< x << endl;
    cout << "Input variances: "<< variances << endl;

    Observation input_observation(x, variances, numberSamples, random_seed);


    double r,s, initX;

    initX = 1.0;
    r = 0.5;
    s = 1.0;


    Data example =  Data(input_observation.x, input_observation.y, r, s, initX, random_seed);
    Trainer trainer(true, SNUV, 100.0, 100, 0.000001);
    trainer.train(example);

    

    return 0;

}