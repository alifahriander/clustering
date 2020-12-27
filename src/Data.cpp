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


    int inSuccess = writeMatrix(x_true.transpose(), "/home/ander/Documents/git/clustering/x_true.csv", 4);
    inSuccess = writeMatrix(y_observed.transpose(), "/home/ander/Documents/git/clustering/y_observed.csv", 4);
    inSuccess = writeMatrix(assignments.transpose(), "/home/ander/Documents/git/clustering/assignments.csv", 4);

    numberClusters = x_true.rows();
    numberSamples = y_observed.rows();
    dimension = x_true.cols();

    forwardMessageW = VectorXd(numberClusters);
    forwardMessageEta =VectorXd(numberClusters);

    // Prepare y 
    y = VectorXd(numberSamples*numberClusters*dimension);
    unsigned int i = 0;
    vector<double> y_flat;

    while(i < y_observed.rows()){
        for(unsigned int j=0; j<numberClusters; j++){
            y_flat.push_back(y_observed(i));
        }
        i++;
    }
    y = VectorXd::Map(y_flat.data(), y_flat.size());

    
    // Prepare A
    A = MatrixXd(numberSamples*numberClusters, numberClusters);
    MatrixXd identity_matrix = MatrixXd::Identity(numberClusters, numberClusters);
    for(unsigned int i = 0; i<numberSamples; ++i){
        A.block(numberClusters*i,0,numberClusters,numberClusters) = identity_matrix;
    }

    // Prepare variances
    r_z = R_z;

    s_z = VectorXd(numberClusters*numberSamples);
    for(unsigned int i=0; i<s_z.rows(); ++i){
        double randomNumber = Data::uniformDistribution(0, 1);
        s_z(i) = randomNumber * randomNumber + r_z*r_z;
    }
    //KMeans++ initialization
    x_estimate = initXEstimate();
    //TODO: initten sonra flattenla 
    // Prepare z
    z = A*x_estimate - y;

    // Data::printData(); 
    Data::saveData(true);
   

}

/*
* Update Rule for W by updating s_x(i)^2 according to (14.25) in lecture notes 
* @param W
* @param s
* @param r: fix initial variance 
*/
void Data::updateW(MatrixXd& W,VectorXd s, double r){
    VectorXd sumVariance = s.array() + r*r;
    sumVariance = sumVariance.array().inverse();
    W = sumVariance.asDiagonal();
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

VectorXd Data::initXEstimate(){
    VectorXd centers(numberClusters);
    // Step 1 : Select one point from y as cluster center
    centers(0) = y_observed((unsigned int)uniformDistribution(0,numberSamples));
    cout << "Center 0:" << centers(0) << endl;
    double prevCenter = centers(0);

    for(unsigned int i = 1; i < numberClusters; i++){
        //Step 2 : Compute distances between 
        VectorXd distances = computeDistances(prevCenter);
        prevCenter = y(selectFromDistribution(distances));
        centers(i) = prevCenter;
        cout << "Center "<< i << ": " << centers(i) << endl;

    }
    return centers;

}

VectorXd Data::computeDistances(double center){
    VectorXd dists(numberSamples);
    for(unsigned int j=0; j<numberSamples; j++){
        dists(j) = (double) pow((y(j) - center),2);
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

void Data::updateData(){
    Data::updateW(W_z, s_z, r_z);
    //TODO: Add updateCost
    // Data::printData();
    Data::saveData(false);
}


void Data::saveData(bool init){
    int success = writeMatrix(x_estimate.transpose(),"/home/ander/Documents/git/clustering/x_estimate.csv", 4);
    if(success != 0) cout<<"Vector couldn't be saved!"<<endl;
    success = writeMatrix(s_x.transpose(),"/home/ander/Documents/git/clustering/s_x.csv", 4);
    if(success != 0) cout<<"Vector couldn't be saved!"<<endl;
    success = writeMatrix(s_z.transpose(),"/home/ander/Documents/git/clustering/s_z.csv", 4);
    if(success != 0) cout<<"Vector couldn't be saved!"<<endl;
    success = writeMatrix(z.transpose(),"/home/ander/Documents/git/clustering/z.csv", 4);
    if(success != 0) cout<<"Vector couldn't be saved!"<<endl;
    if(!init){
    // success = writeMatrix((MatrixXd) costZ,"/home/ander/Documents/git/clustering/costZ.csv", 10);
    // if(success != 0) cout<<"Vector couldn't be saved!"<<endl;
    }
}

