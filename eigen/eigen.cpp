#include "eigen.hpp"

int main()
{
    std::vector<double> blens(4, 1.);
    value_grad ret = vbsky_loglik(blens);
    std::cout << "log(P)=" << ret.log_P << std::endl;
    std::cout << "grad(log(P))=" << ret.grad.transpose() << std::endl;
    return 0;
}
