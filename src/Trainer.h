#ifndef TRAINER_H
#define TRAINER_H
#include <eigen3/Eigen/Core>
#include <iostream>
#include "Data.h"


using namespace std;
using namespace Eigen;

#define IRGD    True
#define IRCD    False

#define SNUV    0
#define HUBER   1
#define L1      2



class Trainer{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        VectorXd gradient;
        bool mode;
        unsigned int numberIterations;
        unsigned int prior;
        double learningRate;
        double tol;
        VectorXd stateX;


        Trainer(bool mode, unsigned int prior, double learningRate, unsigned int numberIterations, double tol);
        void setStateX(Data data);
        void computeGradient(Data data);
        void updateX(Data& data);
        void updateS(Data& data);
        void train(Data& data);

};
#endif