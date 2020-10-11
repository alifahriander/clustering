#include <iostream>
#include <eigen3/Eigen/Core>
#include  <chrono>
#include  <random>
#include <fstream>
#include "Data.h"
#include "Trainer.h"
#include "Observation.h"


using namespace std;
using namespace Eigen;

unsigned random_seed = std::chrono::system_clock::now().time_since_epoch().count();    

int MatrixToCSV(const MatrixXd& inputMatrix, const string& fileName, const streamsize dPrec) {
	int i, j;
	ofstream outputData;
	outputData.open(fileName);
	if (!outputData)
		return -1;
	outputData.precision(dPrec);
    outputData<<",\n";
	for (i = 0; i < inputMatrix.rows(); i++) {
		outputData << inputMatrix(i);
        if(i<inputMatrix.rows()-1) outputData << ",";
        else outputData<<endl;
	}
	outputData << endl;
	outputData.close();

	if (!outputData)
		return -1;
	return 0;
}


int main(){
    // MatrixXd A = MatrixXd::Random(4,6);
    // cout << A << endl;
    // cout << A.col(1) << endl;
    // VectorXd v = A.col(1);
    // v(1) = 2.0;
    // cout<<v<<endl;
    //Training Parameters
    double learning_rate = 0.0001;
    unsigned int numberOfIterations = 1000;
    double tolerance = 0.00001;

    
    unsigned int numberClusters = 2;
    unsigned int numberSamples = 1000;

    // Define cluster centers and variances 
    VectorXd x(numberClusters);
    x << -1.0, 1.0;
    VectorXd variances(numberClusters);
    variances << 0.1, 0.1;

    cout << "Input x: "<< x << endl;
    cout << "Input variances: "<< variances << endl;

    Observation input_observation(x, variances, numberSamples, random_seed);


    double r_x, r_z;

    r_x = 0.5;
    r_z = 0.5;

    Data example =  Data(input_observation, r_x, r_z, random_seed);
    Trainer trainer(IRCD, SNUV, learning_rate, numberOfIterations, tolerance);
    trainer.train(example);

    

    return 0;

}