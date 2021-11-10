#include "reservoir_layer.h"
reservoir_layer::reservoir_layer() {}
reservoir_layer::reservoir_layer(const int unit_size, const double iss_factor, const double input_gain, const double feed_gain, const double p,
	double (*nonlinear)(double), unsigned int seed = 0, const int wash_out = 500) {//変更した
	this->unit_size = unit_size;
	//this->connection_degree = connection_degree;
	this->input_signal_factor = iss_factor;
	//this->weight_factor = weight_factor;
	//this->bias_factor = bias_factor;
	this->p = p;
	this->seed = seed;
	this->nonlinear = nonlinear;
	this->wash_out = wash_out;
	node_type.resize(unit_size + 1);//???????????
	//adjacency_list.resize(unit_size + 1, std::vector<int>(connection_degree + 1));
	weight_reservoir.resize(unit_size + 1, std::vector<double>(connection_degree + 1));
	input_signal_strength.resize(unit_size + 1);
	mt.seed(seed);


	std::vector<std::vector<double>> J; //Jをリサイズしないとダメかも　J.resize(t_size, std::vector<double>(unit_size + 1));
	double pa = 2.0;//曖昧　ここで設定
}

// 結合トポロジーや結合重みなどを設定する  この後マスク信号作るかも｛２値or6値のランダム信号｝→今回は5値ランダム信号
void reservoir_layer::generate_reservoir() {

	std::uniform_real_distribution<> rand_minus1toplus1(-1, 1);//ランダム生成
	std::uniform_int_distribution<> rand_0or1(-2, 2);//intだから0か1

	std::vector<int> permutation(unit_size + 1);      //?？？？？？？？permutation 順列　置換    
	std::iota(permutation.begin(), permutation.end(), 1); //?？？？？　https://kaworu.jpn.org/cpp/std::iota

	for (int n = 1; n <= unit_size; n++) {
		std::shuffle(permutation.begin(), permutation.end(), mt); //?https://cpprefjp.github.io/reference/algorithm/shuffle.html
	}
 

	//各ノードが線形か非線形かを決定
	for (int n = 1; n <= unit_size; n++) {
		if (permutation[n] <= unit_size * p) {  
			node_type[n] = NON_LINEAR;
		}
		else
			node_type[n] = LINEAR;
	}

	for (int n = 1; n <= unit_size; n++) {
		// 入力層の結合重みを決定 マスク信号と入力の強みをここで一緒にしている
		input_signal_strength[n] = input_signal_factor * (rand_0or1(mt) / 2);//最後のなぜこういう式？ あと論文を見る限りinput_signal_factor要素必要？
	}
}


/** リザーバー層を時間発展させる
	 * input_signal 入力信号
	 * output_node[t][n] 時刻tにおけるn番目のノードの出力
	 * t_size ステップ数
	 **/
void reservoir_layer::reservoir_update(const std::vector<double>& input_signal, std::vector<std::vector<double>>& output_node, const int t_size, int seed) {
	std::mt19937 mt2; // メルセンヌ・ツイスタの32ビット版
	mt2.seed(seed);  //???????????????? seedの中rnd()じゃなくていいのか→ランダムだとデバックごとに変わる　（別プロジェクト参照）
	std::uniform_real_distribution<> rand_minus1toplus1(-1, 1);
	output_node[0][0] = 1.0;//変更する要素
	for (int n = 1; n <= unit_size; n++) output_node[0][n] = rand_minus1toplus1(mt2);


	const double e = 2.718281828459045;
	double ξ, d;
	d = 1.0 / (double)unit_size;//分母 +1を消した  d = τ / N→現在τ（遅延時間）を1としているが論文では80としている場合もあった
	ξ = log(1.0 + d);

	std::vector<double> input_sum_node(unit_size + 1, 0);    //要素数unit_size+1、全ての要素の値0 で初期化

	for (int t = 1; t <= t_size; t++) {
		for (int n = 1; n <= unit_size; n++) {
			J[t][n] = input_signal[t - 1] * input_signal_strength[n];
		}
	}

	for (int t = 1; t <= t_size; t++) {//t = 0→t = 1に変更
		output_node[t][0] = output_node[t - 1][unit_size];
		for (int n = 1; n <= unit_size; n++) {
			//output_node[t][n] = activation_function(output_node[t - 1][n], node_type[n], J[t][n]);//ここの引数もっと増えるかも
			output_node[t][n] = activation_function(output_node[t - 1][n], node_type[n]);
			output_node[t][n] *= (1 - pow(e, -ξ));
			output_node[t][n] += pow(e, -ξ) * (output_node[t][n - 1]);
		}
	}
}

void reservoir_layer::reservoir_update_show(const std::vector<double> input_signal, std::vector<std::vector<double>> output_node, const int t_size, const int wash_out, const std::string name) {
	std::uniform_real_distribution<> rand_minus1toplus1(-1, 1);
	output_node[0][0] = 1.0;//変更する要素
	for (int n = 1; n <= unit_size; n++) output_node[0][n] = rand_minus1toplus1(mt);

	std::ofstream outputfile("output_unit/" + name + ".txt");//output_unit 発見！
	outputfile << "t,unit,input,output" << std::endl;

	const double e = 2.718281828459045;
	double ξ, d;
	d = 1.0 / (double)unit_size;//分母 +1を消した
	ξ = log(1.0 + d);

	std::vector<double> input_sum_node(unit_size + 1, 0);    //要素数unit_size+1、全ての要素の値0 で初期化

	for (int t = 1; t <= t_size; t++) {
		for (int n = 1; n <= unit_size; n++) {
			J[t][n] = input_signal[t - 1] * input_signal_strength[n];
		}
	}

	for (int t = 1; t <= t_size; t++) {//t = 0→t = 1に変更
		output_node[t][0] = output_node[t - 1][unit_size];
		for (int n = 1; n <= unit_size; n++) {
			//output_node[t][n] = activation_function(output_node[t - 1][n], node_type[n], J[t][n]);//ここの引数もっと増えるかも
			output_node[t][n] = activation_function(output_node[t - 1][n], node_type[n]);
			output_node[t][n] *= (1 - pow(e, -ξ));
			output_node[t][n] += pow(e, -ξ) * (output_node[t][n - 1]);
		}
		for (int n = 1; n <= unit_size; n++) {
			if (t >= wash_out && t < wash_out + 200)
				outputfile << t << "," << n << "," << input_sum_node[n] << "," << output_node[t + 1][n] << std::endl;
		}
	}
	outputfile.close();
}


bool reservoir_layer::is_echo_state_property(const std::vector<double>& input_signal) {
	auto output_node1 = std::vector<std::vector<double>>(wash_out + 2, std::vector<double>(unit_size + 1, 0));
	auto output_node2 = std::vector<std::vector<double>>(wash_out + 2, std::vector<double>(unit_size + 1, 0));

	reservoir_update(input_signal, output_node1, wash_out, 1);
	reservoir_update(input_signal, output_node2, wash_out, 2);

	double err_sum = 0.0;
	for (int t = wash_out - 99; t <= wash_out; t++) {
		for (int n = 1; n <= unit_size; n++) {
			const double err = (output_node1[t][n] - output_node2[t][n]);
			err_sum += err * err;
		}
	}
	// ノード初期値によって状態が等しくなるならば、EchoStatePropertyを持つ
	double err_ave = err_sum / (unit_size * 10);
	//std::cerr << err_sum << std::endl;
	return err_ave <= 0.1;//△△△
}


//double reservoir_layer::activation_function(const double x, const int type, const double J) {//ここの引数もっと増えるかも
double reservoir_layer::activation_function(const double x, const int type) {
	if (type == LINEAR) {
		return std::max(-1000.0, std::min(1000.0, x));  //?
	}
	else if (type == NON_LINEAR) {
		return nonlinear(x);
		//return nonlinear(x, J, input_gain, feed_gain);
		//もしかすると　nonlinear(x, input_gain, feed_gain, pa, J);なのかも
	}
	assert(type != LINEAR && type != NON_LINEAR);  //?
	return -1.0;
}
