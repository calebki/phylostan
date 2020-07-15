#include "tree.hpp" 

int g_n_tips = 3; 
int g_n_nodes = 2*g_n_tips-1;
int g_n_cols = 2;
pair_mat g_pairs;
vv_p_vec g_tipdata; 
p_vec g_pi;

void on_start(void)
{
    g_tipdata.reserve(g_n_tips);
    for (int i=0; i<g_n_cols; i++) 
    {
        g_tipdata[i].reserve(g_n_cols);   
    }
 
    g_tipdata[0][0] << 0, 0, 0, 1; 
    g_tipdata[0][1] << 0, 0, 0, 1; 
    g_tipdata[0][2] << 1, 0, 0, 0; 
    g_tipdata[0][3] << 1, 0, 0, 0; 
    g_tipdata[0][4] << 0, 1, 0, 0; 
    g_tipdata[1][0] << 0, 0, 1, 0; 
    g_tipdata[1][1] << 0, 1, 0, 0; 
    g_tipdata[1][2] << 0, 1, 0, 0; 
    g_tipdata[1][3] << 0, 1, 0, 0; 
    g_tipdata[1][4] << 0, 1, 0, 0; 

    g_pairs.resize(g_n_nodes, 2);
    g_pairs << 5, -1, 
             4, 5,
             3, 5,
             1, 4,
             2, 4;

    g_pi << 0.25, 0.25, 0.25, 0.25;
}
