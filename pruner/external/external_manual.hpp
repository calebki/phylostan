#include <Eigen/Dense>

inline double vsum (std::vector<double> x, std::ostream* pstream)
{
    double s = 0;
    int n = x.size();
    for (int i=0; i<n; i++)
    {
        s += x[i];
    }
    return s;
}

inline std::vector<double> vsum_gradient (std::vector<double> x, std::ostream* pstream)
{
    int n = x.size();
    std::vector<double> grad(n, 1.0);
    return grad;
}

inline var vsum (const std::vector<var>& x, std::ostream* pstream)
{
    std::vector<double> a = value_of(x);
    double s = vsum(a, pstream);
    std::vector<double> grad = vsum_gradient(a, pstream);
    return precomputed_gradients(s, x, grad);
    // return var(new precomp_v_vari(s, x.vi_, grad));
}