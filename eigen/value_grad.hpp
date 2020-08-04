#pragma once

#include <Eigen/Dense>

struct value_grad {
    double log_P;
    Eigen::VectorXd grad;
};

