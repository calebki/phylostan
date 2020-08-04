#pragma once
#include <Eigen/Dense>
#include "value_grad.hpp"

using std::vector;

extern value_grad bksy_loglik(const vector<double>&);

stan::math::var pruning_loglik(const Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1>& blens, std::ostream*) 
{
    vector<var> blens_v;
    for (int i = 0; i < blens.rows(); ++i) blens_v.push_back(blens(i));
    value_grad ll = vbsky_loglik(value_of(blens_v));
    vector<double> grad;
    for (int i = 0; i < ll.grad.rows(); ++i) grad.push_back(ll.grad(i));
    return precomputed_gradients(ll.log_P, blens_v, grad);
}
