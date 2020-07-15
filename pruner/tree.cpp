#include "tree.hpp"

Node::Node(int k, int p)
{
  key = k;
  visited = false;
  parent = p;

  left = -1;
  right = -1;

  p_partials = Eigen::MatrixXd::Constant(4,g_n_nodes,0);
  q_partials = Eigen::MatrixXd::Constant(4,g_n_nodes,0);
  P = Eigen::MatrixXd::Constant(4,4,0);
}

Node::Node() : Node(-1,-1){}

Node::Node(int k) : Node(k, -1){}

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

void Node::set_p_partial(int col, p_vec p)
{
  p_partials.col(col) = p;
}

void Node::set_q_partial(int col, p_vec q)
{
  q_partials.col(col) = q;
}

void Node::set_P(p_mat X)
{
  P = X;
}

bool Node::get_visited()
{
  return visited;
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

p_vec Node::get_p_partial(int col)
{
  return p_partials.col(col);
}

p_vec Node::get_q_partial(int col)
{
  return q_partials.col(col);
}

p_mat_d Node::get_p_partials()
{
  return p_partials;
}

p_mat_d Node::get_q_partials()
{
  return q_partials;
}

p_mat Node::get_P()
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

Tree::Tree()
{
  on_start();
  root = g_pairs(0,0);
  n_nodes = g_pairs.rows();
  n_tips = (n_nodes + 1) / 2;
  preorder.reserve(n_nodes);
  postorder.reserve(n_nodes);

  preorder.push_back(root);
  postorder.push_back(g_pairs(n_nodes-1,0));
  Node_map[root] = Node(root);

  int child;
  int parent;

  for (int i = 1; i < n_nodes; i++){
    child = g_pairs(i,0);
    parent = g_pairs(i,1);
    Node_map[child] = Node(child, parent);
    Node_map[parent].set_child(child);
    preorder.push_back(child);
    postorder.push_back(g_pairs(n_nodes-1-i,0));
  }
}

void Tree::clear_visited(void)
{
  for (int i = 1; i < n_nodes; i++){
    Node_map[preorder[i]].set_visited(#include <iostream>false);
  }
}

void Tree::clear_visited(int col)
{
  int key;
  Node *curr;
  Node *left;
  Node *right;
  for (int i = 1; i < n_nodes; i++){
    key = postorder[i];
    curr = &Node_map[key];
    if (curr->is_leaf())
    {
      if (g_tipdata[col][key] == g_tipdata[col+1][key])
      {
        curr->set_visited(true);
      }
      else
      {
        curr->set_visited(false);
      }
    }
    else
    {
      left = &Node_map[curr->get_left()];
      right = &Node_map[curr->get_right()];
      if (left->get_visited() && right->get_visited())
      {
        curr->set_visited(true);
      }
      else
      {
        curr->set_visited(false);
      }
    }
  }
}

int Tree::get_other(int key)
{
  Node *curr = &Node_map.at(key);
  Node *parent = &Node_map.at(curr->get_parent());
  Node *left = &Node_map.at(parent->get_left());
  Node *right = &Node_map.at(parent->get_right());
  if (left->get_key() != key)
    return left->get_key();
  else
    return right->get_key();
}

void Tree::calc_Ps(Eigen::VectorXd blens)
{
  int key;
  for (int i = 0; i < n_nodes-1; i++)
  {
    key = preorder[i+1];
    p_mat X = Eigen::MatrixXd::Constant(4,4,0.25 - 0.25*exp(-blens[i]/0.75));
    p_vec d = Eigen::MatrixXd::Constant(4,1,0.25 + 0.75*exp(-blens[i]/0.75));
    X.diagonal() = d;
    Node_map[key].set_P(X);
  }
}

void Tree::calc_p_partials(int col)
{
  int key;
  for (int i=0; i<n_nodes; i++){
    key = postorder[i];
    Node *curr = &Node_map[key];
    if (!curr->get_visited())
    {
      if (curr->is_leaf())
      {
        curr->set_p_partial(col, g_tipdata[col][key]);
      }
      else
      {
        Node *left = &Node_map[curr->get_left()];
        Node *right = &Node_map[curr->get_right()];
        curr->set_p_partial(col, (left->get_P() * left->get_p_partial(col)).cwiseProduct(right->get_P() * right->get_p_partial(col)));
      }
      curr->set_visited(true);
    }
  }
}

void Tree::calc_q_partials(int col)
{
  int key;
  Node *curr = &Node_map[root];
  curr->set_q_partial(col, g_pi);
  //curr->set_visited(true);
  for (int i=1; i<n_nodes; i++){
    key = preorder[i];
    curr = &Node_map[key];
    Node *parent = &Node_map[curr->get_parent()];
    Node *other = &Node_map[get_other(key)];
    curr->set_q_partial(col, curr->get_P().transpose()*(parent->get_q_partial(col).cwiseProduct(other->get_P()*other->get_p_partial(col))));
    //curr->set_visited(true);
  } 
}

std::map<int,Node> Tree::get_Node_map(void)
{
  return Node_map;
}

v_int Tree::get_preorder(void)
{
  return preorder;
}

v_int Tree::get_postorder(void)
{
  return postorder;
}

int Tree::get_root(void)
{
  return root;
}

int Tree::get_n_nodes(void)
{
  return n_nodes;
}

int Tree::get_n_tips(void)
{
  return n_tips;
}

int Tree::get_n_cols(void)
{
  return n_cols;
}

