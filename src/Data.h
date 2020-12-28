#ifndef DATA_H
#define DATA_H

#include <eigen3/Eigen/Core>
#include <iostream>
#include <random>
#include "Observation.h"

using namespace std;
using namespace Eigen;

class Data{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        // unsigned int numberTrueClusters;
        unsigned int numberClusters;
        unsigned int numberExtendedSamples;
        unsigned int numberSamples;

        // Matrix consisting of unit matrix numberSamnumberSamples 
        MatrixXd A;
        // Mean value of cluster points 
        MatrixXd x_true;
        Matrix<double,Dynamic,Dynamic,RowMajor> x_estimateMatrix;
        VectorXd x_estimate;
        // Samples in a matrix 
        MatrixXd Y;

        //y_observed is the observation vector with y_0, y_1, ..., y_N
        VectorXd y_observed;
        // y is the extended observation vector with y_0, y_0, ...,y_0,y_1,... 
        // where every observation is repeated K times 
        VectorXd y;
        VectorXd assignments;
        // z = Ax - y
        VectorXd z;

        // SNUV parameters 
        double r_z;

        // For weighted update 
        long double alpha = 0.1;

        unsigned int dimension;

        VectorXd s_x;
        VectorXd s_z;

        double costX;
        double costZ;

        VectorXd forwardMessageW;
        VectorXd forwardMessageEta;


        default_random_engine generator_data;
    
        Data(Observation inputObservation, double r_z, unsigned random_seed);
        void updateCost(VectorXd v, double r, double& cost);

        void printData();
        void saveData();
        
        Matrix<double,Dynamic,Dynamic,RowMajor> initXEstimate();
        VectorXd computeDistances(VectorXd center);
        unsigned int selectFromDistribution(VectorXd distances);

        double normalDistribution(double mean, double variance);
        double uniformDistribution(double min, double max);

};

#endif
