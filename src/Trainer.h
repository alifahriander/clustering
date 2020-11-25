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
        VectorXd gradient;
        bool mode;
        unsigned int numberIterations;
        unsigned int priorX;
        unsigned int priorZ;
        double learningRate;
        double tol;
        VectorXd stateX;


        Trainer(bool mode, unsigned int priorX, unsigned int priorZ, double learningRate, unsigned int numberIterations, double tol);
        void setStateX(Data data);
        void computeGradient(Data data);
        void updateX(Data& data);
        void updateSx(Data& data);
        void updateSz(Data& data);

        void train(Data& data);
        // int VectorToCSV(const MatrixXd& inputMatrix, const string& fileName, const streamsize dPrec);

};
#endif