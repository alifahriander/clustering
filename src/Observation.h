#ifndef OBSERVATION_H
#define OBSERVATION_H
#include <iostream>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Cholesky>
#include <string>


using namespace Eigen;
using namespace std;

class Observation{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        unsigned int dimension;

        VectorXd x;

        VectorXd assignments;
        
        VectorXd y;
        MatrixXd Y;
        
        VectorXd variances;
        MatrixXd V;

        default_random_engine generator_observation;

        Observation(VectorXd mean_vector, VectorXd variance_vector, unsigned int numberSamples, unsigned seed);
        Observation(MatrixXd mean_matrix, MatrixXd variance_matrix, unsigned int numberSamples, unsigned seed);
        Observation();

        double normalDistribution(double mean, double variance);
        unsigned int uniformDistribution(unsigned int min, unsigned int max);

        void computeObservation(unsigned int numberSamples);
        void computeObservation(MatrixXd meanMatrix, MatrixXd choleskyMatrices[], unsigned int numberSamples);

};
#endif 