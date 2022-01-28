#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() 
{
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
        0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;

  /**
  * Finish initializing the FusionEKF.
  * Set the process and measurement noises
  */
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;

  /**
   * Initialize EKF
   */
  // State transition matrix definition
  ekf_.F_ = MatrixXd(4, 4);

  // Process covariance matrix definition
  ekf_.Q_ = MatrixXd(4, 4);
  
  // State Covariance definition
  ekf_.P_ = MatrixXd(4, 4);
  
  // Initial State Covariance values
  ekf_.P_ << 1, 0, 0, 0,
             0, 1, 0, 0,
             0, 0, 1000, 0,
             0, 0, 0, 1000;
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::Predict(const MeasurementPackage &measurement_pack)
{
  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  /**
  * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
  * Update the process noise covariance matrix.
  * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
  */

  double dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;

  // State transition matrix update
  ekf_.F_ << 1, 0, dt, 0,
             0, 1, 0, dt,
             0, 0, 1, 0,
             0, 0, 0, 1;

  // Noise covariance matrix computation
  // Noise values from the task
  double noise_ax = 9.0;
  double noise_ay = 9.0;

  double dt_pow2 = dt * dt;       //dt^2
  double dt_pow3 = dt_pow2 * dt;  //dt^3
  double dt_pow4 = dt_pow3 * dt;  //dt^4

  double dt_3_over_2 = dt_pow3/2;
  double dt_4_over_4 = dt_pow4/4;

  ekf_.Q_ <<  dt_4_over_4 * noise_ax, 0, dt_3_over_2 * noise_ax, 0,
	            0, dt_4_over_4 * noise_ay, 0, dt_3_over_2 * noise_ay,
	            dt_3_over_2 * noise_ax, 0, dt_pow2 * noise_ax, 0,
 	            0, dt_3_over_2 * noise_ay, 0, dt_pow2 * noise_ay;

  ekf_.Predict();
}

void FusionEKF::Update(const MeasurementPackage &measurement_pack)
{
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
  	ekf_.R_ = R_radar_;
  	ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    // Laser updates
    ekf_.H_ = H_laser_;
  	ekf_.R_ = R_laser_;
  	ekf_.Update(measurement_pack.raw_measurements_);
  }
}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) 
{
  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
    * Initialize the state ekf_.x_ with the first measurement.
    * Create the covariance matrix.
    * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */

    // first measurement
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // Convert radar from polar to cartesian coordinates and initialize state.
      VectorXd radarStates(3);
      radarStates[0] = measurement_pack.raw_measurements_[0];   // rho
      radarStates[1] = measurement_pack.raw_measurements_[1];   // phi
      radarStates[2] = measurement_pack.raw_measurements_[2];   // rho_dot

      // Coordinates convertion from polar to cartesian
      double cos_phi = cos(radarStates[1]);
      double sin_phi = sin(radarStates[1]);
      double px = radarStates[0] * cos_phi;
      CONSTRAIN(px, .0001, px);
      double py = radarStates[0] * sin_phi;
      CONSTRAIN(py, .0001, py);
      double vx = radarStates[2] * cos_phi;
      double vy = radarStates[2] * sin_phi;

      ekf_.x_ << px, py, vx , vy;
    } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      // Initialize state. We save measurements raw.
      ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
    }

    // Saving first timestamp in seconds
    previous_timestamp_ = measurement_pack.timestamp_ ;
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/
  Predict(measurement_pack);

  /*****************************************************************************
   *  Update
   ****************************************************************************/
  Update(measurement_pack);

  // print the output
  std::cout << "x_ = " << ekf_.x_ << std::endl;
  std::cout << "P_ = " << ekf_.P_ << std::endl;
}