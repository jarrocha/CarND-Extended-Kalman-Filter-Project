#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

#ifndef MAX
#define MAX(a,b) (((a)>(b))?(a):(b))
#endif

#ifndef MIN
#define MIN(a,b) (((a)<(b))?(a):(b))
#endif

#ifndef CONSTRAIN
#define CONSTRAIN(a,low,high) MAX((low),MIN((a),(high)))
#endif

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) 
{
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() 
{
  /**
   * Predict State
   */

  x_ = F_ * x_ ;
  P_ = F_*P_*F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) 
{
  /**
  * update the state by using Kalman Filter equations
  */

  VectorXd y = z - H_ * x_;
  UpdateState(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) 
{
  /**
  * update the state by using Extended Kalman Filter equations
  */

  // px, py, vx, vy
  VectorXd x_ekf(4);
  x_ekf << x_(0), x_(1), x_(2), x_(3);

  // Transformation step for state vector to polar coordinates conversion
  double rho = sqrt(x_ekf[0]*x_ekf[0] + x_ekf[1]*x_ekf[1]);
  double theta = atan2(x_ekf[1], x_ekf[0]);
  double rho_dot = (x_ekf[0]*x_ekf[2] + x_ekf[1]*x_ekf[3]) / rho;

  VectorXd h(3);
  h << rho, theta, rho_dot;
  VectorXd y = z - h;

  if (y(1) > M_PI) {
    while (y(1) > M_PI) {
      y(1) -= M_PI;
    }
  } else {
    while (y(1) < -M_PI) {
      y(1) += M_PI;
    }
  }

  UpdateState(y);
}

void KalmanFilter::UpdateState(const VectorXd &y)
{
  MatrixXd S = H_ * P_ * H_.transpose() + R_;
  MatrixXd K = P_ * H_.transpose() * S.inverse();

  // New State
  x_ = x_ + (K * y);
  int x_size = x_.size();
  MatrixXd Identity = MatrixXd::Identity(x_size, x_size);
  P_ = (Identity - K * H_) * P_;
}