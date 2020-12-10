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


VectorXd loadVector(string path){
    ifstream dataFile(path);
    string rowString;
    string entry;
    int rowNumber = 0;

    vector<double> vectorEntries;


    while (getline(dataFile, rowString))
    {
        stringstream rowStringStream(rowString); 
 
        while (getline(rowStringStream, entry, ',')) 
        {
            vectorEntries.push_back(stod(entry));
        }
        rowNumber++; //update the column numbers
    }
    
    return Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(vectorEntries.data(), vectorEntries.size());
}

double loadScalar(string path){
    double value;

    ifstream dataFile(path);
    
    string rowString;
    string entry;

    while (getline(dataFile, rowString)){
        stringstream rowStringStream(rowString); 
        while (getline(rowStringStream, entry, ',')) value = stod(entry);
    }
    return value;
}

void loadVectors(VectorXd& x, VectorXd& y, VectorXd& assignments){
    x = loadVector("data/x.csv");
    y = loadVector("data/y.csv");
    assignments = loadVector("data/assignments.csv");
}

void loadScalars(double& r_z, unsigned int& numberOfIterations, double& tolerance){
    r_z = loadScalar("config/r_z.csv");
    numberOfIterations = (unsigned int) loadScalar("config/numberOfIterations.csv");
    tolerance = loadScalar("config/tolerance.csv");
}


int writeVector(const MatrixXd& inputMatrix, const string& fileName, const streamsize dPrec) {
	int i;
    assert( inputMatrix.cols() == 1 );

	ofstream outputData;
    // Always appends
	outputData.open(fileName, ios::app);
	if (!outputData)
		return -1;
	outputData.precision(dPrec);
	for (i = 0; i < inputMatrix.rows(); i++) {
		outputData << inputMatrix(i);
        outputData << endl;
	}
	outputData.close();

	if (!outputData)
		return -1;
	return 0;
}


int main(int argc, char** argv){
    // Read from command line 
    bool DATA_READY;
    istringstream(argv[1]) >> DATA_READY;

    // Load configuration
    double r_z;
    unsigned int numberIterations;
    double tolerance;
    loadScalars(r_z, numberIterations, tolerance);


    //Load vectors from CSV
    VectorXd x;
    VectorXd y;
    VectorXd assignments;
    Observation inputObservation = Observation();


    if(DATA_READY){
        loadVectors(x, y, assignments);

        inputObservation.y = y;
        inputObservation.assignments = assignments;
        inputObservation.x = x;

    }else{
        // Read config for number of samples, clusters and variances

        VectorXd centers = loadVector("config/centers.csv");
        VectorXd variances = loadVector("config/variances.csv");

        unsigned int numberSamples, numberClusters;
        numberClusters = centers.size();
        numberSamples = loadScalar("config/numberSamples.csv");

        inputObservation = Observation(centers, variances, numberSamples, random_seed);
        writeVector(inputObservation.assignments, "data/assignments.csv", 4);
        writeVector(inputObservation.y, "data/y.csv", 4);
        writeVector(centers,"data/x.csv", 4);

        cout << "Data created and saved in data folder."<< endl;
        cout << "Data consists of " << numberClusters << "clusters and " << numberSamples << " samples."<< endl;
        return 0;
    }

 
    Data inputData =  Data(inputObservation, r_z, random_seed);

    
    // Create config.csv 
    ofstream configData;
    configData.open("config.csv",ios::app);
    configData << "numberOfIterations" << "," << numberIterations << endl;
    configData << "tolerance" << "," << tolerance << endl;
    configData << "r_z" << "," << r_z << endl;
    configData.close();

    Trainer trainer(numberIterations, tolerance);
    trainer.train(inputData);
    

    return 0;

}