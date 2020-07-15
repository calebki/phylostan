#ifndef TREE_H
#define TREE_H
#include <iostream>
#include <map>
#include <vector>
#include <Eigen/Dense>

typedef std::vector<int> v_int;
typedef std::vector<v_int> vv_int;
typedef std::vector<double> v_double;

typedef Eigen::Matrix<double, 4, 1> p_vec;
typedef Eigen::Matrix<double, 4, 4> p_mat;
typedef Eigen::Matrix<double, 4, Eigen::Dynamic> p_mat_d;
typedef Eigen::Matrix<int, Eigen::Dynamic, 2> pair_mat;

typedef std::vector<p_vec> v_p_vec;
typedef std::vector<v_p_vec> vv_p_vec;

extern int g_n_tips;
extern int g_n_nodes; 
extern int g_n_cols;
extern pair_mat g_pairs;
extern vv_p_vec g_tipdata; 
extern p_vec g_pi;

void on_start(void);

class Node
{
  public:
    Node();
    Node(int k);
    Node(int k, int p);

    //Setter methods
    void set_visited(bool v);
    void set_child(int c);
    void set_p_partial(int col, p_vec p);
    void set_q_partial(int col, p_vec q);
    void set_p_partials(p_mat_d ps);
    void set_q_partials(p_mat_d qs);
    void set_P(p_mat X);

    //Getter methods
    bool get_visited(void);
    int get_key();
    int get_left();
    int get_right();
    int get_parent();
    p_vec get_p_partial(int col);
    p_vec get_q_partial(int col);
    p_mat_d get_p_partials();
    p_mat_d get_q_partials();
    p_mat get_P();

    bool is_leaf(void);

  private:
    int key; //Node id
    bool visited; //check if node has been visited
    int parent; 
    int left; //first child
    int right; //second child
    p_mat_d p_partials; //postorder partial vector
    p_mat_d q_partials; //preorder partial vector
    p_mat P; //probability transition matrix
    p_mat Q; //rate matrix
};

class Tree
{
  public:
    Tree();
    void clear_visited(void);
    void clear_visited(int col);
    int get_other(int key);
    void calc_Ps(Eigen::VectorXd blens);
    void calc_p_partials(int col);
    void calc_q_partials(int col);

    // getter methods
    int get_root(void);
    int get_n_nodes(void);
    int get_n_tips(void);
    int get_n_cols(void);
    v_int get_preorder(void);
    v_int get_postorder(void);
    std::map<int,Node> get_Node_map(void);

  private:
    int root;
    int n_tips;
    int n_nodes;
    int n_cols;
    std::map<int,Node> Node_map;
    v_int preorder;
    v_int postorder;
};

#endif