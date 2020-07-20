#include "pruner.hpp"

double loglik (const Eigen::VectorXd &x, std::ostream* pstream)
{

}

std::vector<double> loglik_grad (const Eigen::VectorXd &x, std::ostream* pstream)
{

}

var loglik (const Eigen::Matrix<var, -1, 1>& x, std::ostream* pstream)
{
    Tree tree = Tree();
    for (int i = 0; i < g_n_cols-1; i++)
    {
        tree.calc_p_partials(i);
        tree.calc_q_partials(i);
        t.clear_visited();
    }
    tree.calc_p_partials(g_n_cols-1);
    tree.calc_q_partials(g_n_cols-1);

    const var * x_data = x.data();
    int x_size = x.size();
    std::vector<var> x_std_var(x_data, x_data + x_size );
    Eigen::VectorXd a = value_of(x);

    double s = vsum(a, pstream);
    std::vector<double> grad = vsum_gradient(a, pstream);
    return precomputed_gradients(s, x_std_var, grad);

}