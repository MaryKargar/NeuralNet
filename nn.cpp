#define MLPACK_ENABLE_ANN_SERIALIZATION
#include <mlpack.hpp>

using namespace mlpack;
using namespace ens;
using namespace std;
using namespace arma;

// Function declaration for ComputeMSE.
double ComputeMSE(mat& pred, mat& Y);

double runRegression(int H1, int H2, int H3);

// int main()
// {
//   int H1 = 64;
//   int H2 = 128;
//   int H3 = 64;

//   double mse = runRegression(H1, H2, H3);
//   cout << "Mean Squared Error on Prediction data points: " << mse << endl;

//   return 0;
// }

// Function definition for ComputeMSE.
double ComputeMSE(mat& pred, mat& Y)
{
  return SquaredEuclideanDistance::Evaluate(pred, Y) / (Y.n_elem);
}

double runRegression(int H1, int H2, int H3)
{
  const string datasetPath = "BodyFat.tsv";
  const string modelFile = "nn_regressor.bin";

  constexpr double RATIO = 0.1;
  const int EPOCHS = 100;
  constexpr double STEP_SIZE = 5e-2;
  constexpr int BATCH_SIZE = 32;
  constexpr double STOP_TOLERANCE = 1e-8;

  const bool bTrain = true;
  const bool bLoadAndTrain = false;

  arma::mat dataset;

  bool loadedDataset = data::Load(datasetPath, dataset, true);
  if (!loadedDataset)
    return -1;

  arma::mat trainData, validData;
  data::Split(dataset, trainData, validData, RATIO);

  arma::mat trainX =
      trainData.submat(1, 0, trainData.n_rows - 1, trainData.n_cols - 1);
  arma::mat validX =
      validData.submat(1, 0, validData.n_rows - 1, validData.n_cols - 1);

  arma::mat trainY = trainData.row(0);
  arma::mat validY = validData.row(0);

  data::MinMaxScaler scaleX;
  data::MinMaxScaler scaleY;
  scaleX.Fit(trainX);
  scaleX.Transform(trainX, trainX);
  scaleX.Transform(validX, validX);

  scaleY.Fit(trainY);
  scaleY.Transform(trainY, trainY);
  scaleY.Transform(validY, validY);

  if (bTrain || bLoadAndTrain)
  {
    FFN<MeanSquaredError, HeInitialization> model;
    if (bLoadAndTrain)
    {
      data::Load(modelFile, "NNRegressor", model);
    }
    else
    {
      model.Add<Linear>(H1);
      model.Add<LeakyReLU>();
      model.Add<Linear>(H2);
      model.Add<LeakyReLU>();
      model.Add<Linear>(H3);
      model.Add<LeakyReLU>();
      model.Add<Linear>(1);
    }

    ens::Adam optimizer(
        STEP_SIZE,
        BATCH_SIZE,
        0.9,
        0.999,
        1e-8,
        trainData.n_cols * EPOCHS,
        STOP_TOLERANCE,
        true);

    model.Train(trainX,
                trainY,
                optimizer,
                // ens::PrintLoss(),
                // ens::ProgressBar(),
                ens::EarlyStopAtMinLoss(20));

    data::Save(modelFile, "NNRegressor", model);
  }

  FFN<MeanSquaredError, HeInitialization> modelP;
  data::Load(modelFile, "NNRegressor", modelP);

  arma::mat predOut;
  modelP.Predict(validX, predOut);

  double validMSE = ComputeMSE(validY, predOut);

  scaleY.InverseTransform(predOut, predOut);

  return validMSE;
}
