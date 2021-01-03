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
#include "util.h"

using namespace std;
using namespace Eigen;
unsigned random_seed = std::chrono::system_clock::now().time_since_epoch().count(); 


/**
 * Read scalar value from csv file 
 * @path: Relative path to csv file
 * */
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

/**
 * Read parameters from csv file 
 * @param r_z: allocated variable for r_z
 * @param numberOfIterations: Parameter to determine maximal number of iterations
 * @param tolerance: parameter to determine convergence in order to terminate the algorithm
 * */
void loadScalars(double& r_z, unsigned int& numberOfIterations, double& tolerance){
    r_z = loadScalar("config/r_z.csv");
    numberOfIterations = (unsigned int) loadScalar("config/numberOfIterations.csv");
    tolerance = loadScalar("config/tolerance.csv");
}






int main(int argc, char** argv){
    int success = 0;
    // Read from command line 
    bool DATA_READY;
    istringstream(argv[1]) >> DATA_READY;

    // Load configuration
    double r_z;
    unsigned int numberIterations;
    double tolerance;
    loadScalars(r_z, numberIterations, tolerance);


    //Load vectors from CSV
    MatrixXd x, y, Y, assignments;
    Observation inputObservation = Observation();
    unsigned int dimension;
    cout << "CHECKPOINT 1" << endl;

    if(DATA_READY){
        // If data is already created, read it and initialize corresponding Observation object
        int success = readMatrix(x, "data/x.csv", 4);
        if(success == -1) cout << "Error when reading x.csv" << endl;
        success = readMatrix(y, "data/y.csv", 4);
        if(success == -1) cout << "Error when reading y.csv" << endl;
        success = readMatrix(assignments, "data/assignments.csv", 4);
        if(success == -1) cout << "Error when reading assignments.csv" << endl;
        success = readMatrix(Y, "data/Y.csv", 4);
        if(success == -1) cout << "Error when reading Y.csv" << endl;

        cout << "all matrices are read" << endl;
        cout << " y dimensions : "<< y.rows() << "x"<<y.cols()<<endl;
        inputObservation.y = y;
        inputObservation.x = x;
        inputObservation.assignments = assignments;
        inputObservation.Y = Y;

        dimension = x.cols();
        cout << "Dimension: " << dimension << endl;

    }else{
        //If data is not ready create it with Observation object

        // Read config for number of samples, clusters and variances
        MatrixXd centers, variances;
        success = readMatrix(variances,"config/variances.csv", 4);
        if(success==-1)cout << "variances not read properly" << endl;
        success = readMatrix(centers, "config/centers.csv", 4);
        if(success==-1)cout << "centers not read properly" << endl;
        //TODO: Multiple or one dimensional differentiation
        unsigned int numberSamples, numberClusters;
        numberClusters = centers.rows();
        numberSamples = loadScalar("config/numberSamples.csv");
        dimension = centers.cols();


        inputObservation = Observation(centers, variances, numberSamples, random_seed);
        success = writeMatrix(inputObservation.assignments, "data/assignments.csv", 4);
        if(success==-1)cout << "assignments not written properly" << endl;

        writeMatrix(inputObservation.y, "data/y.csv", 4);
        if(success==-1)cout << "y not written properly" << endl;

        writeMatrix(centers,"data/x.csv", 4);
        if(success==-1)cout << "x not written properly" << endl;
        writeMatrix(inputObservation.Y, "data/Y.csv", 4);
        if(success==-1)cout << "Y Matrix not written properly" << endl;

        cout << "Data created and saved in data folder."<< endl;
        cout << "Data consists of " <<dimension << " dimensions," << numberClusters << " clusters and " << numberSamples << " samples."<< endl;
        return 0;
    }
    

    // Prepare data
    Data inputData =  Data(inputObservation, r_z, random_seed);

    
    // Create config.csv 
    ofstream configData;
    configData.open("config.csv",ios::app);
    configData << "numberOfIterations" << "," << numberIterations << endl;
    configData << "tolerance" << "," << tolerance << endl;
    configData << "r_z" << "," << r_z << endl;
    configData << "dimension" << "," << dimension << endl;
    configData.close();

    // Start training
    Trainer trainer(numberIterations, tolerance);
    trainer.train(inputData);
    

    return 0;

}

