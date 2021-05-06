#include <algorithm>
#include <cassert>
#include <random>
#include <vector>

#define LINEAR (0)
#define NON_LINEAR (1)

class reservoir_layer {
public:
	unsigned int unit_size;										//　ノード数
	unsigned int connection_degree;								//	1ユニット当たりの接続数	(ノード数の1割程度で精度が飽和する）
	double input_signal_strength;						//	入力の強さ
	double weight_factor;								//	ユニット間接続の強さ
	std::vector<std::vector<double>> weight_reservoir;	//  リザーバー層の結合重み
	std::vector<std::vector<int>> adjacency_list;		//  グラフにおける隣接リスト(隣接リスト？→https://qiita.com/drken/items/4a7869c5e304883f539b)
	unsigned int seed;									//	リザーバーの構造を決定するシード値（構造のシードと重みのシードなどの分割をしてもいいかも）
	

};