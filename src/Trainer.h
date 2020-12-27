#ifndef TRAINER_H
#define TRAINER_H
#include <eigen3/Eigen/Core>
#include <iostream>
#include "Data.h"


using namespace std;
using namespace Eigen;

#define IRGD    true
#define IRCD    false

#define SNUV    0
#define HUBER   1
#define L1      2
#define NUV     3



class Trainer{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        unsigned int numberIterations;
        double tol;
        VectorXd stateX;


        Trainer(unsigned int numberIterations, double tol, unsigned int dimension);
        void updateX(Data& data);
        void updateSz(Data& data);
        void train(Data& data);
        // int VectorToCSV(const MatrixXd& inputMatrix, const string& fileName, const streamsize dPrec);

};
#endif