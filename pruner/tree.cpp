#include "tree.hpp"

Node::Node(int k)
{
  key = k;
  visited = false;
  parent = -1;

  left = -1;
  right = -1;

  p_partial = Eigen::MatrixXd::Constant(4,1,0);
  q_partial = Eigen::MatrixXd::Constant(4,1,0);
  P = Eigen::MatrixXd::Constant(4,4,0);
}

Node::Node(int k, int p)
{
  key = k;
  visited = false;
  parent = p;

  left = -1;
  right = -1;

  p_partial = Eigen::MatrixXd::Constant(4,1,0);
  q_partial = Eigen::MatrixXd::Constant(4,1,0);
  P = Eigen::MatrixXd::Constant(4,4,0);
}

void Node::set_visited(bool b)
{
  visited = b;
}

void Node::set_child(int c)
{
  if(left==-1)
    left = c;
  else
    right = c;
}

void Node::set_p_partial(pvec p)
{
  p_partial = p;
}

void Node::set_q_partial(pvec q)
{
  q_partial = q;
}

void Node::set_P(pmat X)
{
  P = X;
}

int Node::get_key()
{
  return key;
}

int Node::get_left()
{
  return left;
}

int Node::get_right()
{
  return right;
}

int Node::get_parent()
{
  return parent;
}

pvec Node::get_p_partial()
{
  return p_partial;
}

pvec Node::get_q_partial()
{
  return q_partial;
}

pmat Node::get_P()
{
  return P;
}

bool Node::is_leaf(void)
{
  if (left == -1 && right == -1)
    return true;
  else
    return false;
}

Tree::Tree(int c_p_pair[][2])
{
  root = c_p_pair[0][0];
  n =  sizeof(c_p_pair) / sizeof(c_p_pair[0]);
  preorder.reserve(n);
  postorder.reserve(n);

  preorder.push_back(root);
  postorder.push_back(c_p_pair[n-1][0]);
  Node_map.insert({root, Node(root)});

  int key;
  int value;

  for (int i = 1; i < n; i++){
    key = c_p_pair[i][0];
    value = c_p_pair[i][1];
    Node_map.insert({key, Node(key, value)});
    preorder.push_back(key);
    postorder.push_back(c_p_pair[n-1-i][0]);
  }
}

void Tree::clear_visited(void)
{
  for (int i = 1; i < n; i++){
    Node_map.at(preorder[i]).set_visited(false);
  }
}

int Tree::get_other(int key)
{
  Node curr = Node_map.at(key);
  Node parent = Node_map.at(curr.get_parent());
  Node left = Node_map.at(parent.get_left());
  Node right = Node_map.at(parent.get_right());
  if (left.get_key() != key)
    return left.get_key();
  else
    return right.get_key();
}

void Tree::calc_Ps(v_double blens)
{
  int key;
  for (int i = 0; i < n-1; i++)
  {
    key = preorder[i+1];
    pmat X = Eigen::MatrixXd::Constant(4,4,0.25 - 0.25*exp(-blens[i]/0.75));
    pvec d = Eigen::MatrixXd::Constant(4,1,0.75 * exp(-blens[i]/0.75));
    X.diagonal() = d;
    Node_map.at(key).set_P(X);
  }
}

void Tree::calc_p_partials(pvec *tipdata)
{
  int key;
  for (int i=0; i<n; i++){
    key = postorder[i];
    Node curr = Node_map[key];
    if (curr.is_leaf())
    {
      curr.set_p_partial(tipdata[n-1-i]);
    }
    else
    {
      Node left = Node_map[curr.get_left()];
      Node right = Node_map[curr.get_right()];
      curr.set_p_partial((left.get_P() * left.get_p_partial()).cwiseProduct(right.get_P() * right.get_p_partial()));
    }
    curr.set_visited(true);
  }
}

void Tree::calc_q_partials(pvec pi)
{
  int key;
  Node curr = Node_map[root];
  curr.set_q_partial(pi);
  curr.set_visited(true);
  for (int i=1; i<n; i++){
    key = preorder[i];
    curr = Node_map[key];
    Node parent = Node_map[curr.get_parent()];
    Node other = Node_map[get_other(key)];
    curr.set_q_partial(curr.get_P().transpose()*(parent.get_q_partial().cwiseProduct(other.get_P()*other.get_p_partial())));
    curr.set_visited(true);
  } 
}