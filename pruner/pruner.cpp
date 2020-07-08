#include <iostream>
#include <vector>
#include <Eigen/Dense>

namespace tree{
#include "tree.hpp"
}

using namespace std;

typedef Eigen::Matrix<double, 4, 1> pvec;
typedef Eigen::Matrix<double, 4, 4> pmat;

pmat* calculate_jc69_p_mat(vector<double> blens){
		int bcount = blens.size();
    pmat* pmats = new pmat[bcount];
      for(int b = 0; b < bcount; b++) {
        pmats[b] = Eigen::MatrixXd::Constant(4,4,0.25 - 0.25*exp(-blens[b]/0.75));
				pvec d = Eigen::MatrixXd::Constant(4,1,0.75*exp(-blens[b]/0.75));
        pmats[b].diagonal() = d;
			}
		return pmats;
	}

double likelihood (pvec freqs, pvec **tipdata, vector<double> blens, int **post, int pre[][2], double *weights, ostream* pstream){
  int total = sizeof(tipdata);
  int L = sizeof(tipdata[0]);
  int S = total / L;
  pmat Q = Eigen::MatrixXd::Constant(4,4,0.25);
  pvec d = Eigen::MatrixXd::Constant(4,1,-0.75);
  Q.diagonal() = d;

  pmat* Ps = calculate_jc69_p_mat(blens);

  pvec p_partials[2*S-1][L];
  pvec q_partials[2*S-1][L]; 

  pvec p_i;
  pvec p_j;

  double lik = 0;

  for(int i; i < L; i++) {
    for(int n; n < S-1) {
			if (post[n][0] <= S){p_i = tipdata[post[n][0]][i];}
      else{p_i = p_partials[post[n][0]][i];}

      if (post[n][1] <= S){p_j = tipdata[post[n][1]][i];}
      else {p_j = p_partials[post[n][1]][i];}

      p_partials[post[n][2]][i] = (Ps[post[n][0]] * p_i).cwiseProduct(Ps[post[n][1]] * p_j);
		}
    lik += p_partials[post[S-1][2]][i].dot(freqs) * weights[i];
	}
  return lik;
}

int main()
{
  vector<double> blens{1,2,3};
  pmat *P = calculate_jc69_p_mat(blens);
  for (int i = 0; i < 3; i++) 
    cout << P[i] << endl;
}