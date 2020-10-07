#include "Trainer.h"
#include "Data.h"

using namespace std;
using namespace Eigen;

Trainer::Trainer(bool inputMode, unsigned int inputPrior, double inputLearningRate, unsigned int inputNumberInterations, double inputTol){
    Trainer::mode = inputMode;
    Trainer::prior = inputPrior;
    Trainer::learningRate = inputLearningRate;
    Trainer::numberIterations = inputNumberInterations;
    Trainer::tol = inputTol;
}
void Trainer::setStateX(Data data){
    Trainer::stateX = data.x_estimate;    
}

void Trainer::computeGradient(Data data){
    Trainer::gradient = data.W_x * data.x_estimate + data.A.transpose() * data.W_z * data.z;
    // cout << " GRADIENT " << endl << gradient << endl;
    // cout << "W_x dimensions: " << data.W_x.rows() << " x " << data.W_x.cols() << endl; 
    // cout << "W_z dimensions: " << data.W_z.rows() << " x " << data.W_z.cols() << endl; 
    // cout << "A dimensions: " << data.A.rows() << " x " << data.A.cols() << endl; 
    // cout << "z dimensions: " << data.z.rows() << endl; 

}

void Trainer::updateX(Data& data){

    data.x_estimate = data.x_estimate - Trainer::learningRate * Trainer::gradient;
    data.z = data.A * data.x_estimate - data.y;


}


void Trainer::updateS(Data& data){

    if(Trainer::prior == SNUV){
        VectorXd diffX = data.x_estimate.array().pow(2);
        diffX -= data.r_x * data.r_x * VectorXd::Constant(data.numberClusters,1.0);

        for(unsigned int i=0; i<data.numberClusters; ++i){
            if(diffX(i) < 0.0) data.s_x(i) = 0;
            else data.s_x(i) = diffX(i);
        }

        VectorXd diffZ = data.z.array().pow(2);

        diffZ -= data.r_z * data.r_z * VectorXd::Constant(data.numberClusters*data.numberSamples,1.0);

        for(unsigned int i=0; i<data.numberClusters*data.numberSamples; ++i){
            if(diffZ(i) < 0.0) data.s_z(i) = 0;
            else data.s_z(i) = diffZ(i);
        }

        data.updateData();
    }

}

void Trainer::train(Data& data){
    for(unsigned int counter=0; counter<Trainer::numberIterations; counter++){
        Trainer::setStateX(data);
        Trainer::computeGradient(data);
        Trainer::updateX(data);
        Trainer::updateS(data);
        cout << " Difference " << endl;
        cout << (stateX-data.x_estimate).norm() << endl;
        if((stateX-data.x_estimate).norm() < Trainer::tol){
            cout << "CONVERGENCE!" << endl;
            break;
        }
    }
    



}

