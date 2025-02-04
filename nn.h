#ifndef NN_FUNCTIONS_H
#define NN_FUNCTIONS_H

#include <mlpack.hpp>
#include <string>

using namespace arma;

// Function declaration for ComputeMSE.
double ComputeMSE(mat& pred, mat& Y);

double runRegression(int H1, int H2, int H3);

#endif // NN_FUNCTIONS_H