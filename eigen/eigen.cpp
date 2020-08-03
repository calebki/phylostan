#include "eigen.hpp"

int main()
{
    std::vector<double> blens(19,1);
    value_grad ret = vbsky_loglik(blens);
    std::cout << ret.log_P;
    return 0;
}

