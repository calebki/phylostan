#include <iostream>
#include <vector>

#include "tree.hpp"

int main () {
  
  // Fake tree data
  int map[5][2] = {{5,-1}, {4,5}, {3,5}, {1,5}, {2,5}};

  Tree t = Tree(map);
  return 0;
}