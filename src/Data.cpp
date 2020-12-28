#include <fstream>
#include <cmath>
#include "Data.h"
#include "assert.h"
#include "Trainer.h"
#include "util.h"
#include <vector>
#include <iostream>
using namespace std;

Data::Data(Observation inputObservation, double R_z, unsigned random_seed){
    generator_data.seed(random_seed);

    // Get numbers of clusters and samples 
    x_true = inputObservation.x;
    y_observed = inputObservation.y;
    assignments = inputObservation.assignments;
    Y = inputObservation.Y;

    int inSuccess = writeMatrix(x_true.transpose(), "/home/ander/Documents/git/clustering/x_true.csv", 4);
    inSuccess = writeMatrix(y_observed.transpose(), "/home/ander/Documents/git/clustering/y_observed.csv", 4);
    inSuccess = writeMatrix(assignments.transpose(), "/home/ander/Documents/git/clustering/assignments.csv", 4);

    numberClusters = x_true.rows();
    //Since y_observed is already a flattened vector of all multidimensional observations, numberSamples equals to dimension*numberOfSamples
    dimension = x_true.cols();
    numberExtendedSamples = y_observed.rows();
    numberSamples = Y.rows();
    cout << "dimension:" << dimension << endl;
    cout << "numberExtendedSamples:" << numberExtendedSamples << endl;
    cout << "numberSamples:" << numberSamples << endl;

    forwardMessageW = VectorXd(numberClusters*dimension);
    forwardMessageEta =VectorXd(numberClusters*dimension);

    // Prepare y 
    y = VectorXd(numberSamples*numberClusters*dimension);

    unsigned int i = 0;
    vector<double> y_flat;
    while(i < y_observed.rows()){
        for(unsigned int j=0; j<numberClusters*dimension; j++){
            y_flat.push_back(y_observed(i));
        }
        i++;
    }
    y = VectorXd::Map(y_flat.data(), y_flat.size());
    cout << "training y:" << endl;
    cout << y << endl;
    
    //KMeans++ initialization
    Matrix<double,Dynamic,Dynamic,RowMajor> x_estimateMatrix = initXEstimate();
    cout << "x_estimateMatrix" << x_estimateMatrix << endl;
    Map<RowVectorXd> tmp(x_estimateMatrix.data(), x_estimateMatrix.size());
    x_estimate = tmp.transpose();
    cout << "x_estimate vector:\n" << x_estimate << endl;


    // Prepare A
    A = MatrixXd(y.rows(), x_estimate.rows());
    MatrixXd identity_matrix = MatrixXd::Identity(numberClusters*dimension, numberClusters*dimension);
    for(unsigned int i = 0; i<numberExtendedSamples; ++i){
        A.block(numberClusters*dimension*i,0,numberClusters*dimension,numberClusters*dimension) = identity_matrix;
    }
    cout << "A:" << A << endl;
    // // Prepare z
    z = A*x_estimate - y;


    // Prepare variances
    r_z = R_z;

    s_z = VectorXd(z.rows());
    for(unsigned int i=0; i<s_z.rows(); ++i){
        double randomNumber = Data::uniformDistribution(0, 1);
        s_z(i) = randomNumber * randomNumber;
    }
    cout << "s_z:" << s_z << endl;
    s_x = VectorXd(x_estimate.rows());
    cout << "s_x:" << s_x << endl;

    // Data::printData(); 
    Data::saveData();
   

}


void Data::updateCost(VectorXd v, double r, double& cost){
    double accumulatedCost = 0.0;
    for(unsigned int i=0; i<v.size(); i++){
        if(v(i)*v(i) < r*r) accumulatedCost += (v(i)*v(i) /(2*r*r)) + log(r);
        else accumulatedCost += log(abs(v(i))) + 0.5;
    }
    
    cost = accumulatedCost;
}



/*
* Draw sample from normal distribution
* @param mean
* @param variance
* @return sample 
*/
double Data::normalDistribution(double mean, double variance){
    normal_distribution<double> distribution(mean, variance);
    return distribution(generator_data);
}
double Data::uniformDistribution(double min, double_t max){
    uniform_real_distribution<double> distrib(min, max); 
    return distrib(generator_data);

}

Matrix<double,Dynamic,Dynamic,RowMajor> Data::initXEstimate(){
    MatrixXd centers(numberClusters, dimension);
    // Step 1 : Select one point from y as cluster center
    centers.row(0) = Y.row((unsigned int)uniformDistribution(0,numberSamples));
    cout << "Center 0:" << centers.row(0) << endl;
    VectorXd prevCenter = centers.row(0);
    cout << "First Center:" << prevCenter << endl;
    for(unsigned int i = 1; i < numberClusters; i++){
        //Step 2 : Compute distances between 
        VectorXd distances = computeDistances(prevCenter);
        prevCenter = Y.row(selectFromDistribution(distances));
        centers.row(i) = prevCenter;
        cout << "Center "<< i << ": " << centers.row(i) << endl;

    }
    cout << "init centers " << centers << endl;
    return centers;

}

VectorXd Data::computeDistances(VectorXd center){
    VectorXd dists(numberSamples);
    for(unsigned int j=0; j<numberSamples; j++){
        cout << "Y.row(j)" << Y.row(j) << endl;
        cout << "center" << center << endl;
        VectorXd distance = Y.row(j)-center.transpose();
        cout << "Distance Vector "<<  distance << endl;
        dists(j) = distance.squaredNorm();
    }
    return dists;

}

unsigned int Data::selectFromDistribution(VectorXd distances){
    distances = distances.normalized();
    // cout << distances << endl;
    double randomValue = uniformDistribution(0,1);
    double runningSum = 0.0;
    for(unsigned int i = 0; i<numberSamples; i++){
        runningSum += distances(i);
        if(runningSum > randomValue){
            return i;
        }
    }
    return 0;
}


void Data::printData(){
    cout << "========================================" << endl;
    cout << "S_x:" << endl << s_x << endl;
    cout << "S_z:" << endl << s_z << endl;
    cout << "x_estimate" << endl << x_estimate << endl;
    cout << "z=(Ax-y):" << endl << z << endl;


}


void Data::saveData(){
    int success = writeMatrix(x_estimate.transpose(),"/home/ander/Documents/git/clustering/x_estimate.csv", 4);
    if(success != 0) cout<<"Vector couldn't be saved!"<<endl;
    // success = writeMatrix(s_x.transpose(),"/home/ander/Documents/git/clustering/s_x.csv", 4);
    // if(success != 0) cout<<"Vector couldn't be saved!"<<endl;
    success = writeMatrix(s_z.transpose(),"/home/ander/Documents/git/clustering/s_z.csv", 4);
    if(success != 0) cout<<"Vector couldn't be saved!"<<endl;
    success = writeMatrix(z.transpose(),"/home/ander/Documents/git/clustering/z.csv", 4);
    if(success != 0) cout<<"Vector couldn't be saved!"<<endl;
    
}

