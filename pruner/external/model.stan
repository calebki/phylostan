functions{
    real vsum(vector x);
    real eexp_log(vector y, real theta, int n){
        real logP;
        logP = -n*log(theta) - vsum(y)/theta;
        return logP;
    }
}
data {
    int<lower=0> n; // sample size
    vector[n] y; // data
}
parameters {
    real<lower=0> theta;
}
model {
    theta ~ gamma(2, 0.1);
    y ~ eexp(theta, n);
}