#include "reservoir_layer.h"
reservoir_layer::reservoir_layer() {}
reservoir_layer::reservoir_layer(const int unit_size, const double iss_factor, const double input_gain, const double feed_gain, const double p,
	double (*nonlinear)(double, double, double, double), unsigned int seed = 0, const int wash_out = 500, const int t_size = 3000) {//変更した
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
	//weight_reservoir.resize(unit_size + 1, std::vector<double>(connection_degree + 1));
	input_signal_strength.resize(unit_size + 1);
	mt.seed(seed);
	this->input_gain = input_gain;
	this->feed_gain = feed_gain;
	//std::vector<std::vector<double>> J; 
	J.resize(t_size + 1, std::vector<double>(unit_size + 1));
	//double ρ = 2.0;//曖昧　ここで設定
	a.resize(6);
	b.resize(2);
	this->j = j;
	//this->j2 = j2;
}





// 結合トポロジーや結合重みなどを設定する.  ｛２値or6値のランダム信号作る｝
void reservoir_layer::generate_reservoir() {
	 
	//std::uniform_real_distribution<> rand_minus1toplus1(-1, 1);//ランダム生成
	//std::uniform_int_distribution<> rand_minus2toplus2(-2, 2);//intだから0か1
	//std::uniform_int_distribution<> rand_minus1orplus1(-10, 10);
	//std::uniform_int_distribution<> rand_0or5(-2, 3);
	a = { -1.0, -0.6, -0.2, 0.2, 0.6, 1.0 };
	b = {-1.0,1.0};

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
	/*for (int n = 1; n <= unit_size; n++) {

		if (b[rand() % b.size()] == 1) {
			node_type[n] = NON_LINEAR;
		}
		else
			node_type[n] = LINEAR;
	}*/

	// 入力層の結合重みを決定 マスク信号と入力の強みをここで一緒にしている
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
	output_node[0][0] = 1.0;//変更する要素
	for (int n = 1; n <= unit_size; n++) output_node[0][n] = (double)rand_minus1toplus1(mt2);

	//std::vector<double> virtual_output_node(unit_size + 1, 0);


	//const double e = 2.7182818;// 2.718281828459045;
	double ξ, d;
	//τ = (double) unit_size * 0.2;
	d = 33.0 / (double)unit_size;//（遅延時間）を1としているが論文では80としている場合も...　　////////////////////////変更要素/////////////////
	ξ = log(1.0 + d);

	/*
	τ = 95 err_ave  0.1345

	11/11(木)
	50.0でnarmaタスクのnmse0.020(ちなみにゲインは入力1.3付近　フィードは0.3付近にしている)
	30で　　　　　　　　　 0.018
	31.4                   0.01645(pa = 2, d_alpha = 0.05; alpha_min = 0.80 )

	気づいたこと
	　input_gainは1～1.3で言い性能がでやすそう
	  feed_gainは0.3付近　
	  paは低めのほうがよさそう(大体2の時性能がいい)
	*/


	//std::vector<double> input_sum_node(unit_size + 1, 0);    //要素数unit_size+1、全ての要素の値0 で初期化

	//マスク信号を加えた最終的な入力信号
	for (int t = 1; t <= t_size; t++) {
		for (int n = 1; n <= unit_size; n++) {
			J[t][n] = input_signal[t - 1] * input_signal_strength[n];
			//if (n <= 10) std::cerr << t << " " << n << " " << J[t][n] << " " << input_signal[t - 1] << " " << input_signal_strength[n] << std::endl;
		}
	}
	//初期値設定
	for (int n = 1; n <= unit_size; n++) {
		J[0][n] = input_signal_strength[n];
		output_node[0][n] = activation_function(output_node[0][n], 0.0, node_type[n], J[0][n]);
		//output_node[0][n] *= (1.0 - pow(e, -ξ));
		//output_node[0][n] *= d / (1.0 + d); /////////////////////////////////////////////////////変更要素//////////////////////
		output_node[0][n] *= (1.0 - exp(-ξ));
		output_node[0][n] += exp(-ξ) * (output_node[0][n - 1]);
		//output_node[0][n] += (1.0 / (1.0 + d)) * (output_node[0][n - 1]);/////////////////////////////////////////変更要素////////////////////
	}

	/*for (int t = 1; t <= t_size; t++) {//t = 0→t = 1に変更
		output_node[t][0] = output_node[t - 1][unit_size];
		for (int n = 1; n <= unit_size; n++) {
			//output_node[t][n] = activation_function(output_node[t - 1][n], node_type[n], J[t][n]);
			output_node[t][n] = activation_function(output_node[t - 1][n], node_type[n]);
			output_node[t][n] *= (1 - pow(e, -ξ));
			output_node[t][n] += pow(e, -ξ) * (output_node[t][n - 1]);
		}
	}*/


		/*//通常の時間遅延システム型時間発展式
	for (int t = 1; t <= t_size; t++) {//t = 0→t = 1に変更
		output_node[t][0] = output_node[t - 1][unit_size];
		for (int n = 1; n <= unit_size; n++) {
			output_node[t][n] = activation_function(output_node[t - 1][n], 0.0, node_type[n], J[t][n]);//ここの引数もっと増えるかも
				//output_node[t][n] = activation_function(output_node[t - 1][n], node_type[n]);
				//if (n == 1) std::cerr << t << " " << output_node[t - 1][n] << " "<< output_node[t][n] << std::endl;
				//output_node[t][n] *= (1.0 - pow(e, -ξ));
			output_node[t][n] *= (1.0 - exp(-ξ));
				//output_node[t][n] *= (d / (1.0 + d));/////////////////////////////////////結構よかった//////////////////
				//if (n == 1) std::cerr << t << " " << output_node[t][n] << std::endl;
				//output_node[t][n] += pow(e, -ξ) * (output_node[t][n - 1]);
			output_node[t][n] += exp(-ξ) * (output_node[t][n - 1]);
				//output_node[t][n] += (1.0 / (1.0 + d)) * (output_node[t][n - 1]);//////////////結構よかった/////////////////////
				//if (n == 1) std::cerr << t << " " << output_node[t][n] << std::endl;
		}
	}*/




		//二次の時間遅延システム型時間発展式
	j = 80;
	for (int t = 1; t <= t_size; t++) {//t = 0→t = 1に変更
		output_node[t][0] = output_node[t - 1][unit_size];
		for (int n = 1; n <= unit_size; n++) {
			if (t == 1) {
				output_node[t][n] = activation_function(output_node[t - 1][n], 0.0, node_type[n], J[t][n]);
				//if (n == 5) std::cerr << t << " " << output_node[t][n] << std::endl;
			}
			else if (t >= 2) {
				if (n >= j + 1) {
					output_node[t][n] = activation_function(output_node[t - 1][n], output_node[t - 1][n - j], node_type[n], J[t][n]);
					//if (n == 5) std::cerr << t << " " << output_node[t][n] << std::endl;
				}
				else {
					output_node[t][n] = activation_function(output_node[t - 1][n], output_node[t - 2][unit_size - j + n], node_type[n], J[t][n]);
					//if (n == 1) std::cerr << t << " " << output_node[t][n] << "_" << output_node[t - 2][unit_size - j + n] << std::endl;
				}

			}
			//if (n == 1) std::cerr << t << " " << output_node[t][n] << std::endl;

			//output_node[t][n] = activation_function(output_node[t - 1][n], node_type[n]);
			//if (n == 1) std::cerr << t << " " << output_node[t - 1][n] << " "<< output_node[t][n] << std::endl;
			//output_node[t][n] *= (1.0 - pow(e, -ξ));
			output_node[t][n] *= (1.0 - exp(-ξ));
			//output_node[t][n] *= (d / (1.0 + d));/////////////////////////////////////結構よかった//////////////////
			//if (n == 1) std::cerr << t << " " << output_node[t][n] << std::endl;
			//output_node[t][n] += pow(e, -ξ) * (output_node[t][n - 1]);
			output_node[t][n] += exp(-ξ) * (output_node[t][n - 1]);
			//output_node[t][n] += (1.0 / (1.0 + d)) * (output_node[t][n - 1]);//////////////結構よかった/////////////////////
			//if (n == 1) std::cerr << t << " " << output_node[t][n] << std::endl;
			//if (n == unit_size) std::cerr << t << " " << output_node[t][n] << std::endl;
		}
	}
}
/*	//三次の時間遅延システム型時間発展式
	j2 = 60;
	for (int t = 1; t <= t_size; t++) {
		output_node[t][0] = output_node[t - 1][unit_size];
		for (int n = 1; n <= unit_size; n++) {
			if (t == 1) {
				output_node[t][n] = activation_function2(output_node[t - 1][n], 0.0, 0.0, node_type[n], J[t][n]);
			}
			else if (t >= 2) {
				if (n >= j + 1) {
					output_node[t][n] = activation_function2(output_node[t - 1][n], output_node[t - 1][n - j], output_node[t - 1][n - j2], node_type[n], J[t][n]);
				}
				else if(n >= j2 + 1 && n < j + 1){
					output_node[t][n] = activation_function2(output_node[t - 1][n], output_node[t - 2][unit_size - j + n], output_node[t - 1][n - j2], node_type[n], J[t][n]);
				}
				else {
					output_node[t][n] = activation_function2(output_node[t - 1][n], output_node[t - 2][unit_size - j + n], output_node[t - 2][unit_size - j2 + n], node_type[n], J[t][n]);
				}
			}
			//output_node[t][n] *= (1.0 - exp(-ξ));
			output_node[t][n] *= (d / (1.0 + d));
			output_node[t][n] += (1.0 / (1.0 + d)) * (output_node[t][n - 1]);
			//output_node[t][n] += exp(-ξ) * (output_node[t][n - 1]);
		}
	}*/


void reservoir_layer::reservoir_update_show(const std::vector<double> input_signal, std::vector<std::vector<double>> output_node, const int t_size, const int wash_out, const std::string name) {
	std::uniform_real_distribution<> rand_minus1toplus1(-1, 1);
	output_node[0][0] = 1.0;//変更する要素
	for (int n = 1; n <= unit_size; n++) output_node[0][n] = (double)rand_minus1toplus1(mt);

	std::ofstream outputfile("output_unit_STDE/" + name + ".txt");//output_unit 発見！
	//outputfile << "t,unit,input,output" << std::endl;
	outputfile << "t,unit,output" << std::endl;

	//const double e = 2.7182818;// 281828459045;
	double ξ, d;
	d = 17.0 / (double)unit_size;//分母 +1を消した　//////////////////////////////////////////変更要素//////////////////
	ξ = log(1.0 + d);

	//std::vector<double> input_sum_node(unit_size + 1, 0);    //要素数unit_size+1、全ての要素の値0 で初期化

	for (int t = 1; t <= t_size; t++) {
		for (int n = 1; n <= unit_size; n++) {
			J[t][n] = input_signal[t - 1] * input_signal_strength[n];
		}
	}

	
	for (int n = 1; n <= unit_size; n++) {
		J[0][n] = input_signal_strength[n];
		output_node[0][n] = activation_function(output_node[0][n], 0.0, node_type[n], J[0][n]);
		//output_node[0][n] *= d / (1.0 + d); 
		output_node[0][n] *= (1.0 - exp(-ξ));
		//output_node[0][n] += (1.0 / (1.0 + d)) * (output_node[0][n - 1]);
		output_node[0][n] += exp(-ξ) * (output_node[0][n - 1]);
	}


	/*//通常の時間遅延システム型時間発展式
	for (int t = 1; t <= t_size; t++) {//t = 0→t = 1に変更
		output_node[t][0] = output_node[t - 1][unit_size];
		for (int n = 1; n <= unit_size; n++) {
			output_node[t][n] = activation_function(output_node[t - 1][n], 0.0, node_type[n], J[t][n]);//ここの引数もっと増えるかも
			//output_node[t][n] = activation_function(output_node[t - 1][n], node_type[n]);
			//output_node[t][n] *= (1.0 - pow(e, -ξ));
			output_node[t][n] *= (1.0 - exp(-ξ));
			//output_node[t][n] *= (d / (1.0 + d));///////////////////////////////////////////////////////////////
			//output_node[t][n] += pow(e, -ξ) * (output_node[t][n - 1]);
			output_node[t][n] += exp(-ξ) * (output_node[t][n - 1]);
			//output_node[t][n] += (1.0 / (1.0 + d)) * (output_node[t][n - 1]);////////////////////////////////////
		}
	}*/
		
	//二次の時間遅延システム型時間発展式
	const int j = 80;
	for (int t = 1; t <= t_size; t++) {//t = 0→t = 1に変更
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
				//ここの引数もっと増えるかも
				//output_node[t][n] = activation_function(output_node[t - 1][n], node_type[n]);
				//if (n == 1) std::cerr << t << " " << output_node[t - 1][n] << " "<< output_node[t][n] << std::endl;
				//output_node[t][n] *= (1.0 - pow(e, -ξ));
		    output_node[t][n] *= (1.0 - exp(-ξ));
			//output_node[t][n] *= (d / (1.0 + d));/////////////////////////////////////結構よかった//////////////////
			//if (n == 1) std::cerr << t << " " << output_node[t][n] << std::endl;
			//output_node[t][n] += pow(e, -ξ) * (output_node[t][n - 1]);
			output_node[t][n] += exp(-ξ) * (output_node[t][n - 1]);
			//output_node[t][n] += (1.0 / (1.0 + d)) * (output_node[t][n - 1]);//////////////結構よかった/////////////////////
			//if (n == 1) std::cerr << t << " " << output_node[t][n] << std::endl;
		}
		for (int n = 1; n <= unit_size; n++) {
			if (t >= wash_out && t < wash_out + 200)
			//outputfile << t << "," << n << "," << input_sum_node[n] << "," << output_node[t + 1][n] << std::endl;
			outputfile << t << "," << n << "," << output_node[t][n] << std::endl;
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
			//if (n == 1) {
			//	std::cerr << t << " " << err << std::endl;
			//}
		}
	}
	
	// ノード初期値によって状態が等しくなるならば、EchoStatePropertyを持つ
	double err_ave = err_sum / (unit_size * 10);
	//std::cout << err_ave << "\n";
	//std::cerr << err_sum << std::endl;
	//std::cerr << input_signal_factor << " " << input_gain << " " << feed_gain << std::endl;
	/*if (unit_size < 40)
		return err_ave <= 0.2;
	else
		return err_ave <= 0.1;//△△△*/
	return err_ave <= 0.2;
}

double reservoir_layer::activation_function(const double x1,const double x2, const int type, const double J) {//ここの引数もっと増えるかも
//double reservoir_layer::activation_function(const double x, const int type) {
	double x;
	x = x1 + x2;

	if (type == LINEAR) {
		return std::max(-1000.0, std::min(1000.0, x)); 
	}
	else if (type == NON_LINEAR) {
		//return nonlinear(x);
		///double makkey(const double x, double J, double input_gain, double feed_gain) {//Mackey_Glass
		//return feed_gain * (x + input_gain * J) / (1.0 + pow(x + input_gain * J, 2.0));//指数ρ = 2-------------------------
		//return feed_gain * sin(x + input_gain * J + 0.7) * sin(x + input_gain * J + 0.7);

		//return feed_gain * pow(sin(x + input_gain * J + 0.3), 2.0);//池田モデル  φ:オフセット位相
		//return feed_gain * exp(-x) * sin(x + input_gain * J);//expsin ρパラメータの調整必要なし 結構良かった
		//}

		return nonlinear(x, J, input_gain, feed_gain);
		//return nonlinear(x, input_gain, feed_gain, J);
	}
	assert(type != LINEAR && type != NON_LINEAR);  //?
	return -1.0;
}

/*
double reservoir_layer::activation_function2(const double x1, const double x2, const double x3, const int type, const double J) {//ここの引数もっと増えるかも
//double reservoir_layer::activation_function(const double x, const int type) {
	double x;
	x = x1 + x2 + x3;

	if (type == LINEAR) {
		return std::max(-1000.0, std::min(1000.0, x));
	}
	else if (type == NON_LINEAR) {
		//return nonlinear(x);
		///double makkey(const double x, double J, double input_gain, double feed_gain) {//Mackey_Glass
		//return feed_gain * (x + input_gain * J) / (1.0 + pow(x + input_gain * J, 2.0));//指数ρ = 2-------------------------
		//return feed_gain * sin(x + input_gain * J + 0.7) * sin(x + input_gain * J + 0.7);

		//return feed_gain * pow(sin(x + input_gain * J + 0.3), 2.0);//池田モデル  φ:オフセット位相
		//return feed_gain * exp(-x) * sin(x + input_gain * J);//expsin ρパラメータの調整必要なし 結構良かった
		//}

		return nonlinear(x, J, input_gain, feed_gain);
		//return nonlinear(x, input_gain, feed_gain, J);
	}
	assert(type != LINEAR && type != NON_LINEAR);  //?
	return -1.0;
}
*/