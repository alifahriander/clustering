#include <fstream>
#include "Trainer.h"
#include "Data.h"

using namespace std;
using namespace Eigen;

Trainer::Trainer(bool inputMode, unsigned int inputPriorX,unsigned int inputPriorZ, double inputLearningRate, unsigned int inputNumberInterations, double inputTol){
    Trainer::mode = inputMode;
    Trainer::priorX = inputPriorX;
    Trainer::priorZ = inputPriorZ;

    Trainer::learningRate = inputLearningRate;
    Trainer::numberIterations = inputNumberInterations;
    Trainer::tol = inputTol;
}
/**
 * Keep the previous x_estimate to determine whether the solution converged
 * */
void Trainer::setStateX(Data data){
    Trainer::stateX = data.x_estimate;    
}
/**
 * Gradient computation according to IRGD in 14.29 in Lecture Notes
 * */
void Trainer::computeGradient(Data data){
    Trainer::gradient = data.W_x * data.x_estimate + data.A.transpose() * data.W_z * data.z;

}

/**
 * Gradient step and update z
 * */
void Trainer::updateX(Data& data){
    if(mode==IRGD){
        data.x_estimate = data.x_estimate - Trainer::learningRate * Trainer::gradient;
        data.z = data.A * data.x_estimate - data.y;
    }
    else if(mode==IRCD){
        if(Trainer::priorX != NUV){
        VectorXd y_hat = VectorXd(data.numberClusters*data.numberSamples);
        VectorXd g = VectorXd(data.numberClusters*data.numberSamples);

        for(unsigned int i=0; i<data.numberClusters;++i){
            data.x_estimate(i) = 0.0;
            y_hat = data.A * data.x_estimate;
            g = (data.W_z * data.A.col(i)) / (data.W_x(i,i) + data.A.col(i).transpose()*data.W_z*data.A.col(i));
            data.x_estimate(i) = g.transpose() * (data.y-y_hat);
        }
        data.z = data.A * data.x_estimate - data.y;
        }else{
            for(int k = 0; k<data.numberClusters; k++){
                long double w_x = 0.0;
                long double eta_x = 0.0;
                long double tmp = 0.0;

                for(int i=0;i<data.numberSamples;i++){
                    tmp = (data.s_z(i*data.numberClusters + k));
                    if(tmp != 0.0)w_x += 1 / tmp ;
                    else continue;
                    // cout << "Variance :" << tmp << endl;
                    // cout << "W_X= " << w_x << endl;
                    eta_x+= (1/tmp) * data.y(i);
                }
                

                // cout << "ETA_X= " << eta_x << endl;
                data.x_estimate(k) = eta_x / (w_x);
                data.s_x(k) = 1/w_x;
                // cout << isinf(data.s_x(k)) << endl;
                // cout << "X_ESTIMATE:" << data.x_estimate(k) << endl;
            }
            data.z = data.A * data.x_estimate - data.y;

            

        }

    }

}

/**
 * Update s according to the rule in 14.25 in Lecture Notes
 * */
void Trainer::updateSx(Data& data){

    if(Trainer::priorX == SNUV){
        VectorXd diffX = data.x_estimate.array().pow(2);
        diffX -= data.r_x * data.r_x * VectorXd::Constant(data.numberClusters,1.0);
        for(unsigned int i=0; i<data.numberClusters; ++i){
            if(diffX(i) < 0.0) data.s_x(i) = 0;
            else data.s_x(i) = diffX(i);
        }
    }
    else if(Trainer::priorX == HUBER){
        for(unsigned int i=0; i<data.numberClusters; ++i){
            if(data.x_estimate(i)<data.beta*data.r_x*data.r_x) data.s_x(i) = 0.0;
            else data.s_x(i) = sqrt((abs(data.x_estimate(i))/data.beta) - (data.r_x*data.r_x));
        }
    }
    else if(Trainer::priorX == L1){
        for(unsigned int i=0; i<data.numberClusters; ++i){
            data.s_x(i) = sqrt(abs(data.x_estimate(i)/data.beta));
        }
    }


}
void Trainer::updateSz(Data& data){
    if(Trainer::priorZ == SNUV){
        VectorXd diffZ = data.z.array().pow(2);
        // cout << diffZ << endl;

        diffZ -= data.r_z * data.r_z * VectorXd::Constant(data.numberClusters*data.numberSamples,1.0);

        for(unsigned int i=0; i<data.numberClusters*data.numberSamples; ++i){

            if(diffZ(i) < 0.0) data.s_z(i) = 0;
            else data.s_z(i) = diffZ(i);
        }
    }
    else if(Trainer::priorZ == L1){
        for(unsigned int i=0; i<data.numberClusters*data.numberSamples; ++i){
            data.s_z(i) = sqrt(abs(data.z(i)/data.beta));
        }
    }
}

/**
 * Training Loop
 * */
void Trainer::train(Data& data){
    if(mode==IRGD){
        for(unsigned int counter=0; counter<Trainer::numberIterations; counter++){
            Trainer::setStateX(data);
            Trainer::computeGradient(data);
            Trainer::updateX(data);
            Trainer::updateSx(data);
            Trainer::updateSz(data);
            data.updateCost(data.x_estimate,data.r_x,Trainer::priorX,data.costX);
            data.updateCost(data.z,data.r_z,Trainer::priorZ,data.costZ);

            data.updateData();
            cout << " Iteration: "<< counter << "\tDifference: "<< (stateX-data.x_estimate).norm() << endl;

            if((stateX-data.x_estimate).norm() < Trainer::tol){
                cout << "CONVERGENCE!" << endl;
                break;
            }
        }
    }
    else if(mode==IRCD){
        double diff = 0.0;
        unsigned int noChangeCounter = 0;
        for(unsigned int counter=0; counter<Trainer::numberIterations; counter++){
            Trainer::setStateX(data);

            Trainer::updateX(data);
            
            Trainer::updateSx(data);
            Trainer::updateSz(data);
            // Trainer::updateX(data);



            
            
            

            //If at least one of s_xi is 0, stop the training
            bool stopTraining = false;
            for(unsigned int i =0; i<data.numberClusters;i++){
                if(data.s_x(i) == 0.0 ||isinf(data.s_x(i))){
                    stopTraining = true;
                }
            }

            diff = (stateX-data.x_estimate).norm(); 
            cout << " Iteration: "<< counter << "\tDifference: "<< diff << endl;
            
            if(stopTraining) break;

            data.updateCost(data.x_estimate,data.r_x,Trainer::priorX,data.costX);
            data.updateCost(data.z,data.r_z,Trainer::priorZ,data.costZ);
            data.updateData();



            if(diff< Trainer::tol){
                noChangeCounter++;
                if(noChangeCounter == 10){
                    cout << "CONVERGENCE!" << endl;
                    break;
                }
                
            }

        }
    }
}
