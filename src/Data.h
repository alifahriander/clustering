#ifndef DATA_H
#define DATA_H

#include <eigen3/Eigen/Core>
#include <iostream>
#include <random>

using namespace std;
using namespace Eigen;

class Data{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        unsigned int numberClusters;
        unsigned int numberSamples;

        // Matrix consisting of unit matrix numberSamnumberSamples 
        MatrixXd A;
        MatrixXd W_x;
        MatrixXd W_z;


        // Mean value of cluster points 
        VectorXd x_true;
        VectorXd x_estimate;

        //y_observed is the observation vector with y_0, y_1, ..., y_N
        VectorXd y_observed;
        // y is the extended observation vector with y_0, y_0, ...,y_0,y_1,... 
        // where every observation is repeated K times 
        VectorXd y;

        // z = Ax - y
        MatrixXd z;

        // SNUV parameters 
        double r_x;
        double r_z;

        VectorXd s_x;
        VectorXd s_z;

        default_random_engine generator_data;
    
        Data(VectorXd x_true, VectorXd y_observed, double r, double s, double initX, unsigned random_seed);
        void updateW(MatrixXd& W,VectorXd s, double r);
        void updateData();
        double normalDistribution(double mean, double variance);


};

#endif //DATA_H
