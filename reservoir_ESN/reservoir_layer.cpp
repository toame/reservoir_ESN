#include "reservoir_layer.h"
reservoir_layer::reservoir_layer() {}
reservoir_layer::reservoir_layer(const int unit_size, const double iss_factor, const double input_gain, const double feed_gain, const double p,
	double (*nonlinear)(double, double, double, double), unsigned int seed = 0, const int wash_out = 500, const int t_size = 3000) {
	this->unit_size = unit_size;
	this->input_signal_factor = iss_factor;
	this->p = p;
	this->seed = seed;
	this->nonlinear = nonlinear;
	this->wash_out = wash_out;
	node_type.resize(unit_size + 1);
	input_signal_strength.resize(unit_size + 1);
	mt.seed(seed);
	this->input_gain = input_gain;
	this->feed_gain = feed_gain;
	J.resize(t_size + 1, std::vector<double>(unit_size + 1));
	a.resize(6);
	b.resize(2);
	this->j = j;
}



//マスキング処理など
void reservoir_layer::generate_reservoir() {
	 
	//std::uniform_real_distribution<> rand_minus1toplus1(-1, 1);//ランダム生成
	//std::uniform_int_distribution<> rand_minus2toplus2(-2, 2);//intだから0か1
	//std::uniform_int_distribution<> rand_minus1orplus1(-10, 10);
	//std::uniform_int_distribution<> rand_0or5(-2, 3);
	a = { -1.0, -0.6, -0.2, 0.2, 0.6, 1.0 };
	b = {-1.0,1.0};

	std::vector<int> permutation(unit_size + 1);          
	std::iota(permutation.begin(), permutation.end(), 1); 

	for (int n = 1; n <= unit_size; n++) {
		std::shuffle(permutation.begin(), permutation.end(), mt);
	}
	

	//各ノードが線形か非線形かを決定
	for (int n = 1; n <= unit_size; n++) {
		if (permutation[n] <= unit_size * p) {  
			node_type[n] = NON_LINEAR;
		}
		else
			node_type[n] = LINEAR;
	}

	//マスク信号と入力の強みをここで一緒にしている
	for (int n = 1; n <= unit_size; n++) {
		//input_signal_strength[n] = input_signal_factor * (double)(rand_minus1orplus1(mt) / 2.0);
		input_signal_strength[n] = input_signal_factor * a[rand() % a.size()];
		//input_signal_strength[n] = input_signal_factor * b[rand() % b.size()];///////////////////////////変更要素//////////////////////
	}
}



/** リザーバー層を時間発展させる
	 * input_signal 入力信号
	 * output_node[t][n] 時刻tにおけるn番目のノードの出力
	 * t_size ステップ数
	 **/

void reservoir_layer::reservoir_update(const std::vector<double>& input_signal, std::vector<std::vector<double>>& output_node, const int t_size, int seed) {
	std::mt19937 mt2; // メルセンヌ・ツイスタの32ビット版
	mt2.seed(seed);
	std::uniform_real_distribution<> rand_minus1toplus1(-1, 1);
	double exp(double x);
	output_node[0][0] = 1.0;
	for (int n = 1; n <= unit_size; n++) output_node[0][n] = (double)rand_minus1toplus1(mt2);

	double ξ, d;
	d = 2.0 / (double)unit_size;/////////////////////////////////////////////////////////変更要素////////////////////////////////////////////////////////////////////
	ξ = log(1.0 + d);

	//マスク信号を加えた最終的な入力信号
	for (int t = 1; t <= t_size; t++) {
		for (int n = 1; n <= unit_size; n++) {
			J[t][n] = input_signal[t - 1] * input_signal_strength[n];
		}
	}

	//初期値設定
	for (int n = 1; n <= unit_size; n++) {
		J[0][n] = input_signal_strength[n];
		output_node[0][n] = activation_function(output_node[0][n], 0.0, node_type[n], J[0][n]);

		//output_node[0][n] *= d / (1.0 + d); /////////////////////////変更要素//////////////////////
		output_node[0][n] *= (1.0 - exp(-ξ));///////////////////////////////////////////////////////

		//output_node[0][n] += (1.0 / (1.0 + d)) * (output_node[0][n - 1]);//////////////////////変更要素////////////////////
		output_node[0][n] += exp(-ξ) * (output_node[0][n - 1]);/////////////////////////////////////////////////////////////
	}

	//通常の時間遅延システム型時間発展式
	for (int t = 1; t <= t_size; t++) {
		output_node[t][0] = output_node[t - 1][unit_size];
		for (int n = 1; n <= unit_size; n++) {
			output_node[t][n] = activation_function(output_node[t - 1][n], 0.0, node_type[n], J[t][n]);

			//output_node[t][n] *= (d / (1.0 + d));//////////////変更要素//////////////
			output_node[t][n] *= (1.0 - exp(-ξ));/////////////////////////////////////
			
			//output_node[t][n] += (1.0 / (1.0 + d)) * (output_node[t][n - 1]);//////////変更要素////////////
			output_node[t][n] += exp(-ξ) * (output_node[t][n - 1]);/////////////////////////////////////////
		}
	}
}




		/*//二次の時間遅延システム型時間発展式
	j = 60;
	for (int t = 1; t <= t_size; t++) {
		output_node[t][0] = output_node[t - 1][unit_size];
		for (int n = 1; n <= unit_size; n++) {
			if (t == 1) {
				output_node[t][n] = activation_function(output_node[t - 1][n], 0.0, node_type[n], J[t][n]);
			}
			else if (t >= 2) {
				if (n >= j + 1) {
					output_node[t][n] = activation_function(output_node[t - 1][n], output_node[t - 1][n - j], node_type[n], J[t][n]);
				}
				else {
					output_node[t][n] = activation_function(output_node[t - 1][n], output_node[t - 2][unit_size - j + n], node_type[n], J[t][n]);
				}
			}
			//output_node[t][n] *= (d / (1.0 + d));////////////変更要素//////////////
			output_node[t][n] *= (1.0 - exp(-ξ));///////////////////////////////////
			
			//output_node[t][n] += (1.0 / (1.0 + d)) * (output_node[t][n - 1]);///////////変更要素///////////////
			output_node[t][n] += exp(-ξ) * (output_node[t][n - 1]);/////////////////////////////////////////////
		}
	}
}*/


void reservoir_layer::reservoir_update_show(const std::vector<double> input_signal, std::vector<std::vector<double>> output_node, const int t_size, const int wash_out, const std::string name) {
	std::uniform_real_distribution<> rand_minus1toplus1(-1, 1);
	output_node[0][0] = 1.0;
	for (int n = 1; n <= unit_size; n++) output_node[0][n] = (double)rand_minus1toplus1(mt);

	std::ofstream outputfile("output_unit_STDE/" + name + ".txt");
	outputfile << "t,unit,output" << std::endl;
	double ξ, d;
	d = 2.0 / (double)unit_size;//////////////////////変更要素//////////////////
	ξ = log(1.0 + d);

	//マスク信号を加えた最終的な入力信号
	for (int t = 1; t <= t_size; t++) {
		for (int n = 1; n <= unit_size; n++) {
			J[t][n] = input_signal[t - 1] * input_signal_strength[n];
		}
	}

	//初期値設定
	for (int n = 1; n <= unit_size; n++) {
		J[0][n] = input_signal_strength[n];
		output_node[0][n] = activation_function(output_node[0][n], 0.0, node_type[n], J[0][n]);

		//output_node[0][n] *= d / (1.0 + d); /////////////////////////変更要素//////////////////////
		output_node[0][n] *= (1.0 - exp(-ξ));///////////////////////////////////////////////////////

		//output_node[0][n] += (1.0 / (1.0 + d)) * (output_node[0][n - 1]);//////////////////////変更要素////////////////////
		output_node[0][n] += exp(-ξ) * (output_node[0][n - 1]);/////////////////////////////////////////////////////////////
	}


	//通常の時間遅延システム型時間発展式
	for (int t = 1; t <= t_size; t++) {
		output_node[t][0] = output_node[t - 1][unit_size];
		for (int n = 1; n <= unit_size; n++) {
			output_node[t][n] = activation_function(output_node[t - 1][n], 0.0, node_type[n], J[t][n]);

			//output_node[t][n] *= (d / (1.0 + d));//////////////変更要素//////////////
			output_node[t][n] *= (1.0 - exp(-ξ));/////////////////////////////////////

			//output_node[t][n] += (1.0 / (1.0 + d)) * (output_node[t][n - 1]);//////////変更要素////////////
			output_node[t][n] += exp(-ξ) * (output_node[t][n - 1]);/////////////////////////////////////////
		}
	}
}


	/*//二次の時間遅延システム型時間発展式
	const int j = 60;
	for (int t = 1; t <= t_size; t++) {
		output_node[t][0] = output_node[t - 1][unit_size];
		for (int n = 1; n <= unit_size; n++) {
			if (t == 1) 
				output_node[t][n] = activation_function(output_node[t - 1][n], 0.0, node_type[n], J[t][n]);
			else{
				if (n >= j + 1)
					output_node[t][n] = activation_function(output_node[t - 1][n], output_node[t - 1][n - j], node_type[n], J[t][n]);
				else
					output_node[t][n] = activation_function(output_node[t - 1][n], output_node[t - 2][unit_size - j + n], node_type[n], J[t][n]);
			}
			//output_node[t][n] *= (d / (1.0 + d));////////////変更要素//////////////
			output_node[t][n] *= (1.0 - exp(-ξ));///////////////////////////////////

			//output_node[t][n] += (1.0 / (1.0 + d)) * (output_node[t][n - 1]);///////////変更要素///////////////
			output_node[t][n] += exp(-ξ) * (output_node[t][n - 1]);/////////////////////////////////////////////
		}
		for (int n = 1; n <= unit_size; n++) {
			if (t >= wash_out && t < wash_out + 200)
			outputfile << t << "," << n << "," << output_node[t][n] << std::endl;
		}
	}
	outputfile.close();
}*/


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
	//std::cout << err_ave << "\n";
	//std::cerr << err_sum << std::endl;
	//std::cerr << input_signal_factor << " " << input_gain << " " << feed_gain << std::endl;
	if (unit_size < 40)
		return err_ave <= 0.2;
	else
		return err_ave <= 0.1;
	//return err_ave <= 0.1;
}

double reservoir_layer::activation_function(const double x1,const double x2, const int type, const double J) {
	double x;
    x = (x1 + x2) / 2.0;
	//x = x1 + x2;

	if (type == LINEAR) {
		return std::max(-1000.0, std::min(1000.0, x)); 
	}
	else if (type == NON_LINEAR) {
		return nonlinear(x, J, input_gain, feed_gain);
	}
	assert(type != LINEAR && type != NON_LINEAR);
	return -1.0;
}