#include <fstream>
#include <sstream>
#include <string>
#include <eigen3/Eigen/Core>
#include <iostream>


using namespace std;
using namespace Eigen;

int writeMatrix(const MatrixXd& inputMatrix, const string& fileName, const streamsize dPrec);
int readMatrix(MatrixXd& outputMatrix, const string& fileName, const streamsize dPrec);
