#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <cblas.h>
#include <chrono>
#include <random>
#include "reservoir_layer.h"
#include "output_learning.h"
#include "task.h"
#define PHASE_NUM (3)
#define TRAIN (0)//聞いた　　ただ0～2を文字で分かりやすくしただけ
#define VAL (1)
#define TEST (2)
#define MAX_NODE_SIZE (500)
//非線形カーネル　関数の選択　いまのところマッキーグラスのみを想定
double mackey(const double x, double J, double input_gain, double feed_gain) {//Mackey_Glass
	return (feed_gain * (x + input_gain * J)) / (1 + pow(x + input_gain * J, 2));//pa = 2-------------------------
}


//double Mackey_Grass(std::vector<std::vector<double>>& output_node, double input_gain, double feed_gain, double pa, std::vector<std::vector<double>>& J) {

//}
//↑l 155の部分変える必要あるか？？　もしくはこうしなくてもできるか？？ 下で値を決めたinput_gain, feed_gainやreservoir_layer.hで定義したpa,Jを引数にするには

/*double m(const double x) {
	return (feed_gain * (x + input_gain * J)) / (1 + pow(x + input_gain * J, pa));
}*/

/*double sinc(const double x) {
	if (x == 0) return 1.0;
	return sin(PI * x) / (PI * x);
}*/


#include <sstream>

template <typename T>

std::string to_string_with_precision(const T a_value, const int n = 6)//
{
	std::ostringstream out;
	out.precision(n);
	out << std::fixed << a_value;
	return out.str();
}
typedef void (*FUNC)();
int main(void) {
	//std::cout << "成功１" << "\n";
	const int TRIAL_NUM = 3;	// ループ回数 constが付くと変数は書き換えができなくなり、読み取り専用となります。
	const int step = 3000;
	const int wash_out = 500; 
	std::vector<int> unit_sizes = { 400 };

	std::vector<std::string> task_names = { "narma"};
	if (unit_sizes.size() != task_names.size()) return 0;
	std::vector<int> param1 = { 5 };
	std::vector<double> param2 = { 0};
	if (param1.size() != param2.size()) return 0;
	const int alpha_step = 11;
	const int sigma_step = 11;
	std::string task_name;
	std::string function_name;
	//std::cout << "成功２" << "\n";
	//std::cout << "成功３" << "\n";
	std::vector<std::vector<std::vector<std::vector<double>>>> output_node(alpha_step * sigma_step, std::vector<std::vector<std::vector<double>>>(PHASE_NUM, std::vector<std::vector<double>>(step + 2, std::vector<double>(MAX_NODE_SIZE + 1, 0))));
	//std::cout << "成功4" << "\n";
	std::vector<reservoir_layer> reservoir_layer_v(alpha_step * sigma_step);
	std::vector<bool> is_echo_state_property(alpha_step * sigma_step);
	std::vector<std::vector<std::vector<double>>> w(alpha_step * sigma_step, std::vector<std::vector<double>>(10)); // 各リザーバーの出力重み
	std::vector<std::vector<double>> nmse(alpha_step * sigma_step, std::vector<double>(10));	// 各リザーバーのnmseを格納
	//std::cout << "成功5" << "\n";
	for (int r = 0; r < unit_sizes.size(); r++) {
		//std::cout << "成功6" << "\n";
		const int unit_size = unit_sizes[r];
		const std::string task_name = task_names[r];
		//std::cout << "成功7" << "\n";
		std::vector<std::vector<double>> input_signal(PHASE_NUM), teacher_signal(PHASE_NUM);//この２つそれぞれが3種類の配列を持ってるということ？

		std::vector<std::string> function_names = {"mackey"};//適宜他の
		double alpha_min, d_alpha;//タスクによって最小値が変わる　
		double sigma_min, d_sigma;
		double d_bias;
		std::ofstream outputfile("output_data/" + task_name + "_" + std::to_string(param1[r]) + "_" + to_string_with_precision(param2[r], 1) + "_" + std::to_string(unit_size) + ".txt");
		// 入力信号 教師信号の生成
		//std::cout << "成功8" << "\n";
		for (int phase = 0; phase < PHASE_NUM; phase++) {//論文　手順１
			
			if (task_name == "narma") {
				d_bias = 0.4;
				//d_alpha = 0.05; alpha_min = 0.10; 現状これ
				d_alpha = 0.05; alpha_min = 0.80;//τ = 30の時こっちのほうが良い性能だった
				d_sigma = 0.07; sigma_min = 0.5;
				const int tau = param1[r];
				generate_input_signal_random(input_signal[phase], -1.0, 2.0, step, phase + 1);
				generate_narma_task(input_signal[phase], teacher_signal[phase], tau, step);
				//std::cout << "成功9" << "\n";
			}
			// 入力分布[-1, 1] -> 出力分布[0, 0.5]のnarmaタスク
			else if (task_name == "narma2") {
				d_bias = 0.4;
				d_alpha = 0.005; alpha_min = 0.002;
				d_sigma = 0.07; sigma_min = 0.5;
				const int tau = param1[r];
				generate_input_signal_random(input_signal[phase], -1.0, 2.0, step, phase + 1);
				generate_narma_task2(input_signal[phase], teacher_signal[phase], tau, step);
			}
			else if (task_name == "henon") {
				d_bias = 1.0;
				d_alpha = 0.2; alpha_min = 0.1;
				d_sigma = 0.04; sigma_min = 0.04;
				const int fstep = param1[r];
				generate_henom_map_task(input_signal[phase], teacher_signal[phase], fstep, step, phase * step);
			}
			else if (task_name == "laser") {
				//std::cout << "成功10" << "\n";
				d_bias = 0.5;
				d_alpha = 0.4; alpha_min = 0.1;
				d_sigma = 0.1; sigma_min = 0.1;
				const int fstep = param1[r];
				//std::cout << "成功11" << "\n";
				generate_laser_task(input_signal[phase], teacher_signal[phase], fstep, step, phase * step);
				//std::cout << "成功12" << "\n";
			}
			else if (task_name == "approx") {
				const int tau = param1[r];
				const double nu = param2[r];
				if (tau == 7) { d_alpha = 1.0; alpha_min = 0.1; d_bias = 0.5; d_sigma = 0.03; sigma_min = 0.1; }
				else if (tau == 5) { d_alpha = 2.0; alpha_min = 0.5; d_bias = 1.0; d_sigma = 0.02; sigma_min = 0.02; }
				else if (tau == 3) { d_alpha = 5.0; alpha_min = 1.0; d_bias = 4.0;  d_sigma = 0.02; sigma_min = 0.02; }
				else if (tau == 1) {
					d_alpha = 10.0; alpha_min = 1.0; d_bias = 20.0;  d_sigma = 0.02; sigma_min = 0.02;
				}
				else {
					std::cerr << "error! approx parameter is not setting" << std::endl;
					return 0;
				}

				generate_input_signal_random(input_signal[phase], -1.0, 2.0, step, phase + 1);
				task_for_function_approximation(input_signal[phase], teacher_signal[phase], nu, tau, step, phase);
			}
			else if (task_name == "legendre") {
				const int tau = param1[r];
				const double nu = param2[r];
				if (tau == 7) { d_alpha = 1.0; alpha_min = 0.1; d_bias = 0.5; d_sigma = 0.03; sigma_min = 0.1; }
				else if (tau == 5) { d_alpha = 2.0; alpha_min = 0.5; d_bias = 1.0; d_sigma = 0.02; sigma_min = 0.02; }
				else if (tau == 3) { d_alpha = 5.0; alpha_min = 1.0; d_bias = 4.0;  d_sigma = 0.02; sigma_min = 0.02; }
				else if (tau == 0) {
					d_alpha = 30.0; alpha_min = 100.0; d_bias = 6.0;  d_sigma = 0.01; sigma_min = 0.00;
				}
				else {
					std::cerr << "error! legendre parameter is not setting" << std::endl;
					return 0;
				}

				generate_input_signal_random(input_signal[phase], -1.0, 2.0, step, phase + 1);
				generate_legendre_task(input_signal[phase], teacher_signal[phase], nu, tau, step);
			}
		}
		//std::cout << "成功10" << "\n";
		// 設定出力
		outputfile << "### task_name: " << task_name << std::endl;
		outputfile << "### " << param1[r] << " " << param2[r] << std::endl;
		outputfile << "### input_signal_factor [" << alpha_min << ", " << alpha_min + d_alpha * (alpha_step - 1) << "]" << std::endl;
		//outputfile << "### weight_factor [0.1, 1.1]" << std::endl;
		//outputfile << "function_name,seed,unit_size,p,input_singal_factor,bias_factor,weight_factor,lm,train_nmse,nmse,test_nmse" << std::endl;
		outputfile << "function_name,seed,unit_size,p,input_singal_factor,input_gain,feed_gain,lm,train_nmse,nmse,test_nmse" << std::endl;
		//-------------------------------------------------------------------------------------------------------------------------------------------------------
		std::chrono::system_clock::time_point  start, end; // 型は auto で可
		for (auto function_name : function_names) {
			//double (*nonlinear)(double);//変更
			double (*nonlinear)(double, double, double, double);
			if (function_name == "mackey") nonlinear = mackey;
			//else if (function_name == "tanh") nonlinear = tanh;
			//else if (function_name == "gauss") nonlinear = gauss;
			//else if (function_name == "oddsinc") nonlinear = oddsinc;
			//else if (function_name == "sinc") nonlinear = sinc;
			else {
				std::cerr << "error! " << function_name << "is not found" << std::endl;
				return 0;
			}

			for (int loop = 0; loop < TRIAL_NUM; loop++) {//論文 p12 ばらつき低減
				for (int ite_p = 9; ite_p <= 9; ite_p += 1) {//論文　手順２
					const double p = ite_p * 0.1;
					double opt_nmse = 1e+10;//opt 最適な値  ここでは基準を作っている。 l233あたりで書き換えのコードがある。
					double opt_input_signal_factor = 0;
					//double opt_bias_factor = 0;
					//double opt_weight_factor = 0;
					double opt_feed_gain = 0;//最適なフィードバックゲイン
					double opt_input_gain = 0;//最適な入力ゲイン
					double opt_lm2 = 0;//lmはλのこと
					double test_nmse = 1e+10;
					double train_nmse = 1e+10;
					reservoir_layer opt_reservoir_layer;
					std::vector<double> opt_w;
					start = std::chrono::system_clock::now(); // 計測開始時間
					//std::cout << "成功11" << "\n";

					for (int ite_input = 1; ite_input <= 10; ite_input += 1) {//入力ゲイン(τ = 95 pa = 2 ノード100の時は 1～1.3付近で最適なリザバーが出来上がっていた(あと、NARMAタスク, d_bias = 0.4 d_alpha = 0.05, d_sigma = 0.07))
						//const double input_gain = d_bias * ite_input * 0.1;//d_biasの部分たぶん無くす　
						const double input_gain = 1.2 +  ite_input * 0.04;
						//const double input_gain = 0.2 + ite_input * 0.03;
						for (int ite_feed = 1; ite_feed <= 10; ite_feed += 1) {//τ = 95 pa = 2 ノード100の時は 0.35で最適なリザバーが出来上がることが多かった
							//const double feed_gain = d_bias * ite_feed / 20.0;//d_biasの部分無くす、もしくは変更する--  フィードバックゲインパラメーターηを1から3の間で変化させます。すでに説明したように、自律領域のTDRは、これらのパラメーター値に対して、±（η- 1）1/2;
							const double feed_gain = 0.65 + ite_feed * 0.01;
							//const double feed_gain = 1.0 + ite_feed * 0.1;
#pragma omp parallel for num_threads(32)//ここも変えないとダメ
						// 複数のリザーバーの時間発展をまとめて処理
							for (int k = 0; k < alpha_step; k++) {
								//std::cout << "成功12" << "\n";
								const double input_signal_factor = k * d_alpha + alpha_min;//なぜこの計算なのか？
								//const double weight_factor = (k % sigma_step) * d_sigma + sigma_min;

								//reservoir_layer reservoir_layer1(unit_size, unit_size / 10, input_signal_factor, weight_factor, bias_factor, p, nonlinear, loop, wash_out);
								reservoir_layer reservoir_layer1(unit_size, input_signal_factor, input_gain, feed_gain, p, nonlinear, loop, wash_out, step);
								
								reservoir_layer1.generate_reservoir();
								//std::cout << "成功13" << "\n";
								reservoir_layer1.reservoir_update(input_signal[TRAIN], output_node[k][TRAIN], step);//論文　手順３　　　TRAINつまり0のものを引数にしている　？？？→l66でoutput_nodeを4次元の配列？を定義　　北村さんに確認（この行のoutput_nodeとreservoir_layerのoutput_nodeの引数。前者は
								//std::cout << "成功14" << "\n";
								reservoir_layer1.reservoir_update(input_signal[VAL], output_node[k][VAL], step);//??論文　手順５　　一つ上のupdateを上書きするということではない？
								is_echo_state_property[k] = reservoir_layer1.is_echo_state_property(input_signal[VAL]);
								reservoir_layer_v[k] = reservoir_layer1;//??
							}
							int lm;

							int opt_k = 0;

							output_learning output_learning[341];//？？

#pragma omp parallel for  private(lm) num_threads(32)//??
							// 重みの学習を行う
							for (int k = 0; k < alpha_step; k++) {
								//std::cout << "成功15" << "\n";
								//if (!is_echo_state_property[k]) continue;     //　https://www.comp.sd.tmu.ac.jp/spacelab/c_lec2/node61.html
								//std::cout << "成功16" << "\n";

								output_learning[k].generate_simultaneous_linear_equationsA(output_node[k][TRAIN], wash_out, step, unit_size);
								//std::cout << "成功17" << "\n";
								output_learning[k].generate_simultaneous_linear_equationsb(output_node[k][TRAIN], teacher_signal[TRAIN], wash_out, step, unit_size);

								double opt_lm = 0;
								double opt_lm_nmse = 1e+9;
								
								for (lm = 0; lm < 10; lm++) {
									for (int j = 0; j <= unit_size; j++) {
										output_learning[k].A[j][j] += pow(10, -15 + lm);//べき乗　10の-15+lm乗　論文の式(18)より
										if (lm != 0) output_learning[k].A[j][j] -= pow(10, -16 + lm);
									}
									output_learning[k].IncompleteCholeskyDecomp2(unit_size + 1);
									double eps = 1e-12;
									int itr = 10;
									output_learning[k].ICCGSolver(unit_size + 1, itr, eps);
									w[k][lm] = output_learning[k].w;//おそらく論文　手順４　　　　??????????????? [k][lm]→ある入力強み、ユニット間強みの中の、あるλの場合の重み
									nmse[k][lm] = calc_nmse(teacher_signal[VAL], output_learning[k].w, output_node[k][VAL], unit_size, wash_out, step, false);//論文　手順5続き（論文の書き方がややこしくなってるけど、S1(t)を使って求めた重みとS2(t)を使って予測値を計算で求めてNMSEを出そう！！というだけだと思う。）
								}
								
							}

							// 検証データでもっとも性能の良いリザーバーを選択
							for (int k = 0; k < alpha_step; k++) {//論文　手順６
								//if (!is_echo_state_property[k]) continue;
								for (int lm = 0; lm < 10; lm++) {
									if (nmse[k][lm] < opt_nmse) {
										opt_nmse = nmse[k][lm];
										opt_input_signal_factor = d_alpha * k + alpha_min;
										//opt_bias_factor = bias_factor;
										//opt_weight_factor = (k % sigma_step) * d_sigma + sigma_min;
										opt_feed_gain = feed_gain;
										opt_input_gain = input_gain;
										opt_lm2 = lm;
										opt_k = k;
										opt_w = w[k][lm];
										opt_reservoir_layer = reservoir_layer_v[k];
										train_nmse = calc_nmse(teacher_signal[TRAIN], opt_w, output_node[opt_k][TRAIN], unit_size, wash_out, step, false);
										//std::cerr << train_nmse << " " << opt_input_signal_factor << " " << opt_feed_gain << " " << opt_input_gain << std::endl;
									}
								}

							}
							//std::cout << "成功18" << "\n";
						}
						//std::cout << "成功19" << "\n";
					}
					//std::cout << "成功20" << "\n";

					/*** TEST phase ***/  //論文　手順7
					std::string output_name = task_name + "_" + std::to_string(param1[r]) + "_" + to_string_with_precision(param2[r], 1) + "_" + function_name + "_" + std::to_string(unit_size) + "_" + std::to_string(loop) + "_" + std::to_string(ite_p);

					std::vector<std::vector<double>> output_node_test(step + 2, std::vector<double>(MAX_NODE_SIZE + 1, 0));// △　+2とか MAX_NODE_SIZEとか
					opt_reservoir_layer.reservoir_update(input_signal[TEST], output_node_test, step);
					//std::cout << "成功21" << "\n";
					test_nmse = calc_nmse(teacher_signal[TEST], opt_w, output_node_test, unit_size, wash_out, step, true, output_name);//l241と引数の数違うけど...
					//std::cout << "成功22" << "\n";
					end = std::chrono::system_clock::now();  // 計測終了時間
					double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); //処理に要した時間をミリ秒に変換

					outputfile << function_name << "," << loop << "," << unit_size << "," << std::fixed << std::setprecision(4) << p << "," << opt_input_signal_factor << "," << opt_input_gain << "," << opt_feed_gain << "," << opt_lm2 << "," << std::fixed << std::setprecision(8) << train_nmse << "," << opt_nmse << "," << test_nmse << std::endl;
					std::cerr << function_name << "," << loop << "," << unit_size << "," << std::fixed << std::setprecision(3) << p << "," << opt_input_signal_factor << "," << opt_input_gain << "," << opt_feed_gain << "," << opt_lm2 << "," << std::setprecision(5) << train_nmse << "," << opt_nmse << "," << test_nmse << " " << elapsed / 1000.0 << std::endl;

					// リザーバーのユニット入出力を表示
					opt_reservoir_layer.reservoir_update_show(input_signal[TEST], output_node_test, step, wash_out, output_name);//「output_unit」
					//if (test_nmse > 2.0) {
					//	for (int k = 0; k < alpha_step * sigma_step; k++) {
					//		if (!is_echo_state_property[k]) continue;
					//		for (int lm = 0; lm < 10; lm++) {
					//			std::cerr << k << " " << lm << " " << nmse[k][lm] << std::endl;
					//		}
					//	}
					//}

				}

			}
		}
		outputfile.close();
	}
}






