#include <Eigen/Dense>

// inline double vsum (std::vector<double> x, std::ostream* pstream)
// {
//     double s = 0;
//     int n = x.size();
//     for (int i=0; i<n; i++)
//     {
//         s += x[i];
//     }
//     return s;
// }

inline double vsum (const Eigen::VectorXd& x, std::ostream* pstream)
{
    return x.sum();
}

inline vector<double> vsum_gradient (const Eigen::VectorXd& x, std::ostream* pstream)
{
    int n = x.size();
    std::vector<double> grad(n, 1.0);
    return grad;
}

inline var vsum (const Eigen::Matrix<var, -1, 1>& x, std::ostream* pstream)
{
    const var * x_data = x.data();
    int x_size = x.size();
    vector<var> x_std_var(x_data, x_data + x_size );
    Eigen::VectorXd a = value_of(x);

    double s = vsum(a, pstream);
    vector<double> grad = vsum_gradient(a, pstream);
    return precomputed_gradients(s, x_std_var, grad);
    
    // return var(new precomp_v_vari(s, x.vi_, grad));
}