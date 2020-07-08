#ifndef TREE_H
#define TREE_H
#include <map>
#include <vector>
#include <Eigen/Dense>

typedef std::vector<int> v_int;
typedef std::vector<v_int> vv_int;
typedef std::vector<double> v_double;

typedef Eigen::Matrix<double, 4, 1> pvec;
typedef Eigen::Matrix<double, 4, 4> pmat;

class Node
{
  public:
    Node(int k);
    Node(int k, int p);
    ~Node();

    //Setter methods
    void set_visited(bool v);
    void set_child(int c);
    void set_p_partial(pvec p);
    void set_q_partial(pvec q);
    void set_P(pmat X);

    //Getter methods
    int get_key();
    int get_left();
    int get_right();
    int get_parent();
    pvec get_p_partial();
    pvec get_q_partial();
    pmat get_P();

    bool is_leaf(void);

  private:
    int key;
    bool visited;
    int parent;
    int left;
    int right;
    pvec p_partial;
    pvec q_partial;
    pmat P;
};

class Tree
{
  public:
    Tree(int c_p_pair[][2]);
    ~Tree();
    void clear_visited();
    void calc_Ps(v_double blens);
    void calc_p_partials(pvec *tipdata);
    void calc_q_partials(pvec pi);
    int get_other(int key);

  private:
    int root;
    int n;
    std::map<int,Node> Node_map;
    v_int preorder;
    v_int postorder;
};

#endif