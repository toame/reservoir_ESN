#include <iostream>
#include <algorithm>
#include <cassert>
#include <random>
#include <vector>
#include <numeric>
#include <cassert>
#include <fstream>
#include <string>
#define LINEAR (0)
#define NON_LINEAR (1)

class reservoir_layer {
public:
	unsigned int unit_size;								//　ノード数
	unsigned int connection_degree;						//	1ユニット当たりの接続数	(ノード数の1割程度で精度が飽和する）
	double input_signal_factor;							//	入力の強さ
	double weight_factor;								//	ユニット間接続の強さ
	std::vector<std::vector<double>> weight_reservoir;	//  リザーバー層の結合重み
	std::vector<std::vector<int>> adjacency_list;		//  グラフにおける隣接リスト(隣接リスト:https://qiita.com/drken/items/4a7869c5e304883f539b)
	std::vector<double> input_signal_strength;			//  入力層の重み結合の強さベクトル
	unsigned int seed;									//	リザーバーの構造を決定するシード値（構造のシードと重みのシードなどの分割をしてもいいかも）
	double p;											//	ノードで使用される活性化関数の非線形の割合
	double (*nonlinear)(double);						//	非線形関数の関数ポインタ
	std::vector<int> node_type;							//	n番目のノードの線形/非線形の種類
	std::mt19937 mt;
	int wash_out;

	reservoir_layer();
	reservoir_layer(const int unit_size, const int connection_degree, const double iss_factor, const double weight_factor, const double p,
		double (*nonlinear)(double), const unsigned int seed, const int wash_out);

	void generate_reservoir();
	void reservoir_update(const std::vector<double>& input_signal, std::vector<std::vector<double>>& output_node, const int t_size, int seed = 0);
	void reservoir_update_show(const std::vector<double> input_signal, std::vector<std::vector<double>> output_node, const int t_size, const int wash_out, const std::string name);
	bool is_echo_state_property(const std::vector<double>& input_signal);
	double activation_function(const double x, const int type);

};