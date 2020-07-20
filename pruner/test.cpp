#include <iostream>
#include <vector>

#include "tree.hpp"

void print(std::vector <int> const &a) {
   std::cout << "The vector elements are : ";

   for(int i=0; i < a.size(); i++)
    std::cout << a.at(i) << ' ';
   std::cout << std::endl;
}

int main () {

  Eigen::VectorXd blens;
  blens.resize(g_n_nodes-1);
  blens << 1,2,1,1;

  Tree t = Tree();
  t.calc_Ps(blens);
  print(t.get_postorder());
  t.calc_p_partials(0);
  t.calc_q_partials(0);
  t.clear_visited(0)  ;
  for(auto elem : t.get_Node_map())
  {
    std::cout << "Node: " << elem.first << std::endl;
    std::cout << elem.second.get_visited() << std::endl;
  }
  t.calc_p_partials(1);
  t.calc_q_partials(1);


  for(auto elem : t.get_Node_map())
  {
    std::cout << "Node: " << elem.first << std::endl;
    std::cout << "P matrix:" << std::endl;
    std::cout << elem.second.get_P() << std::endl;
    std::cout << "p partial:" << std::endl;
    std::cout << elem.second.get_p_partials() << std::endl;
    std::cout << "q partial:" << std::endl;
    std::cout << elem.second.get_q_partials() << std::endl; 
    std::cout << "lik: " << elem.second.get_p_partials().cwiseProduct(elem.second.get_q_partials()).colwise().sum() << std::endl; 
  }
  return 0;
}