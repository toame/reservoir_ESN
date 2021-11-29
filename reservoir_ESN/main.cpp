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
#define TRAIN (0)//　ただ0～2を文字で分かりやすくしただけ
#define VAL (1)
#define TEST (2)
#define MAX_NODE_SIZE (500)
//非線形カーネル　関数の選択　いまのところマッキーグラスのみを想定
double STDE_MG(const double x, double J, double input_gain, double feed_gain) {//Mackey_Glass
	return (feed_gain * (x + input_gain * J)) / (1.0 + pow(x + input_gain * J, 2.0));//ρ = 2-------------------------
}
double STDE_ikeda(const double x, double J, double input_gain, double feed_gain) {
	return feed_gain * pow(sin(x + input_gain * J + 0.35), 2.0);
}

double tanh(const double x, double J, double input_gain, double feed_gain) {
	return feed_gain * tanh(x + input_gain * J);
}

double sinc(const double x, double J, double input_gain, double feed_gain) {
	if (x == 0) return 1.0;
	return feed_gain * (sin(PI * (x + input_gain * J)) / (PI * (x + input_gain * J)));
}

double STDE_exp(const double x, double J, double input_gain, double feed_gain) {
	return feed_gain * exp(-x) * sin(x + input_gain * J);
}


//double Mackey_Grass(std::vector<std::vector<double>>& output_node, double input_gain, double feed_gain, double pa, std::vector<std::vector<double>>& J) {

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
	const int TRIAL_NUM = 3;	
	const int step = 3000;
	const int wash_out = 500; 
	std::vector<int> unit_sizes = { 300 };

	std::vector<std::string> task_names = { "narma"};
	if (unit_sizes.size() != task_names.size()) return 0;
	std::vector<int> param1 = { 10 };
	std::vector<double> param2 = { 0.0};
	if (param1.size() != param2.size()) return 0;
	const int alpha_step = 11;
	const int sigma_step = 11;
	std::string task_name;
	std::string function_name;
	std::vector<std::vector<std::vector<std::vector<double>>>> output_node(alpha_step * sigma_step, std::vector<std::vector<std::vector<double>>>(PHASE_NUM, std::vector<std::vector<double>>(step + 2, std::vector<double>(MAX_NODE_SIZE + 1, 0))));
	std::vector<reservoir_layer> reservoir_layer_v(alpha_step * sigma_step);
	std::vector<bool> is_echo_state_property(alpha_step * sigma_step);
	std::vector<std::vector<std::vector<double>>> w(alpha_step/* * sigma_step, */,std::vector<std::vector<double>>(10)); // 各リザーバーの出力重み
	std::vector<std::vector<double>> nmse(alpha_step * sigma_step, std::vector<double>(10));	// 各リザーバーのnmseを格納
	for (int r = 0; r < unit_sizes.size(); r++) {
		const int unit_size = unit_sizes[r];
		const std::string task_name = task_names[r];
		std::vector<std::vector<double>> input_signal(PHASE_NUM), teacher_signal(PHASE_NUM);

		std::vector<std::string> function_names = { "STDE_MG", "STDE_ikeda", };// "STDE_MG", "STDE_ikeda",      "STDE_exp",                          };//  "sinc"は時間あれば
		double alpha_min, d_alpha;//タスクによって最小値が変わる　
		double sigma_min, d_sigma;
		double d_bias;
		std::ofstream outputfile("output_data_STDE/" + task_name + "_" + std::to_string(param1[r]) + "_" + to_string_with_precision(param2[r], 1) + "_" + std::to_string(unit_size) + ".txt");
		// 入力信号 教師信号の生成
		for (int phase = 0; phase < PHASE_NUM; phase++) {//論文　手順１
			
			if (task_name == "narma") {
				d_bias = 0.2;
				//d_alpha = 0.05; alpha_min = 0.10; 現状これ(NARMA10も含めると)
				//d_alpha = 0.05; alpha_min = 0.80;NARMA5に限ってはこっち
				d_alpha = 0.02; alpha_min = 0.4;
				//d_alpha = 0.02; alpha_min = 0.4;
				d_sigma = 0.07; sigma_min = 0.4;
				const int tau = param1[r];
				generate_input_signal_random(input_signal[phase], -1.0, 2.0, step, phase + 1);
				generate_narma_task(input_signal[phase], teacher_signal[phase], tau, step);
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
				/*int a = 0;
				auto function_name = function_names;
				if (function_name[a] == "TDE_MG") {
						d_alpha = 0.2; alpha_min = 0.0;
				}
				else if (function_name[a] == "TDE_ikeda") {
					d_alpha = 2.0; alpha_min = 0.0;
				}*/


				//d_bias = 1.0;
				//d_alpha = 0.2; alpha_min = 0.0;
				//d_sigma = 0.04; sigma_min = 0.04;
				const int fstep = param1[r];
				generate_henom_map_task(input_signal[phase], teacher_signal[phase], fstep, step, phase * step);
			}
			else if (task_name == "laser") {//実データに近い  サンタフェ
				//このタスクでは、サンタフェ時系列競争の混沌とした時系列の一歩先の予測をおこなう
				// とくにカオス領域で動作する赤外線レーザーによって作成されたデータセット1万ポイントの継続ファイルを使用
				d_bias = 0.5;
				//d_alpha = 2.0; alpha_min = 0.1;
				d_sigma = 0.1; sigma_min = 0.1;
				const int fstep = param1[r];
				generate_laser_task(input_signal[phase], teacher_signal[phase], fstep, step, phase * step);
			}
			else if (task_name == "approx") {
				const int tau = param1[r];
				const double nu = param2[r];
				/*if (tau == 7) { d_alpha = 1.0; alpha_min = 0.1; d_bias = 0.5; d_sigma = 0.03; sigma_min = 0.1; }
				else if (tau == 5) { d_alpha = 2.0; alpha_min = 0.5; d_bias = 1.0; d_sigma = 0.02; sigma_min = 0.02; }
				else if (tau == 3) { d_alpha = 5.0; alpha_min = 1.0; d_bias = 4.0;  d_sigma = 0.02; sigma_min = 0.02; }
				else if (tau == 1) {
					d_alpha = 10.0; alpha_min = 1.0; d_bias = 20.0;  d_sigma = 0.02; sigma_min = 0.02;
				}
				else {
					std::cerr << "error! approx parameter is not setting" << std::endl;
					return 0;
				}*/

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

		// 設定出力
		outputfile << "### task_name: " << task_name << std::endl;
		outputfile << "### " << param1[r] << " " << param2[r] << std::endl;
		outputfile << "### input_signal_factor [" << alpha_min << ", " << alpha_min + d_alpha * (alpha_step - 1) << "]" << std::endl;
		//outputfile << "### weight_factor [0.1, 1.1]" << std::endl;
		//outputfile << "function_name,seed,unit_size,p,input_singal_factor,bias_factor,weight_factor,lm,train_nmse,nmse,test_nmse" << std::endl;
		outputfile << "function_name,seed,unit_size,p,input_singal_factor,input_gain,feed_gain,lm,train_nmse,nmse,test_nmse,test_nrmse" << std::endl;
		//-------------------------------------------------------------------------------------------------------------------------------------------------------
		std::chrono::system_clock::time_point  start, end; // 型は auto で可
		for (auto function_name : function_names) {
			//double (*nonlinear)(double);//変更
			double (*nonlinear)(double, double, double, double);
			if (function_name == "STDE_MG") {
				nonlinear = STDE_MG;
				//d_alpha = 0.05; alpha_min = 0.1;
			}
			else if (function_name == "tanh") {
				//d_alpha = 0.2; alpha_min = 15.0;
				nonlinear = tanh;
			}
			//else if (function_name == "gauss") nonlinear = gauss;
			//else if (function_name == "oddsinc") nonlinear = oddsinc;
			else if (function_name == "sinc") nonlinear = sinc;
			else if (function_name == "STDE_ikeda") {
				nonlinear = STDE_ikeda;
				//d_alpha = 0.05; alpha_min = 0.1;
			}
			else if (function_name == "STDE_exp") {
				nonlinear = STDE_exp;
				//d_alpha = 0.05; alpha_min = 0.1;
			}
			else {
				std::cerr << "error! " << function_name << "is not found" << std::endl;
				return 0;
			}

			for (int loop = 0; loop < TRIAL_NUM; loop++) {//論文 p12 ばらつき低減
				for (int ite_p = 8; ite_p <= 10; ite_p += 1) {//論文　手順２
					const double p = ite_p * 0.1;
					double opt_nmse = 1e+10;//opt 最適な値  
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
					//std::ofstream outputfile2("nmse_gain_data/" + task_name + "_" + std::to_string(param1[r]) + "_" + to_string_with_precision(param2[r], 1) + "_" + std::to_string(unit_size) + ".txt");
					//outputfile2 << "function_name,input_gain,feed_gain,opt_nmse" << std::endl;
					for (int ite_input = 1; ite_input <= 10; ite_input += 1) {//入力ゲイン(τ = 95 pa = 2 ノード100の時は 1～1.3付近で最適なリザバーが出来上がっていた(あと、NARMAタスク, d_bias = 0.4 d_alpha = 0.05, d_sigma = 0.07))
						//const double input_gain = d_bias * ite_input * 0.1;//d_biasの部分たぶん無くす　
						//const double input_gain = 0.8 + ite_input * 0.05;
						//NARMA10の場合300秒かけた結果、入力ゲインが0.25, フィードゲインが0.9の時に0.16418というNMSEを達成
						//const double input_gain = 0.2 + ite_input * 0.02;
						//const double input_gain = 0.1 + ite_input * 0.1;
						//const double input_gain = 0.5 + ite_input * 0.05;
						const double input_gain = 0.05 + ite_input * 0.0;
						//const double input_gain = 0.7 + ite_input * 0.04;
						for (int ite_feed = 1; ite_feed <= 10; ite_feed += 1) {//τ = 95 pa = 2 ノード100の時は 0.35で最適なリザバーが出来上がることが多かった
							//double opt_nmse = 1e+10;
							//const double feed_gain = d_bias * ite_feed / 20.0;//d_biasの部分無くす、もしくは変更する--  フィードバックゲインパラメーターηを1から3の間で変化させます。すでに説明したように、自律領域のTDRは、これらのパラメーター値に対して、±（η- 1）1/2;
							//const double feed_gain = 0.72 + ite_feed * 0.04;
							//const double feed_gain = 0.5 + ite_feed * 0.05;
						    //const double feed_gain = 0.8 + ite_feed * 0.04;
							//const double feed_gain = 0.3 + ite_feed * 0.02;
							const double feed_gain = 0.75 + ite_feed * 0.02;
#pragma omp parallel for num_threads(32)
						// 複数のリザーバーの時間発展をまとめて処理
							for (int k = 0; k < alpha_step; k++) {
								const double input_signal_factor = k * d_alpha + alpha_min;
								//const double weight_factor = (k % sigma_step) * d_sigma + sigma_min;

								//reservoir_layer reservoir_layer1(unit_size, unit_size / 10, input_signal_factor, weight_factor, bias_factor, p, nonlinear, loop, wash_out);
								reservoir_layer reservoir_layer1(unit_size, input_signal_factor, input_gain, feed_gain, p, nonlinear, loop, wash_out, step);
								
								reservoir_layer1.generate_reservoir();
								reservoir_layer1.reservoir_update(input_signal[TRAIN], output_node[k][TRAIN], step);//論文　手順３　
								//std::cout << "成功" << "\n";

								reservoir_layer1.reservoir_update(input_signal[VAL], output_node[k][VAL], step);//??論文　手順５　
								is_echo_state_property[k] = reservoir_layer1.is_echo_state_property(input_signal[VAL]);
								reservoir_layer_v[k] = reservoir_layer1;//??
							}
							int lm;

							int opt_k = 0;

							output_learning output_learning[341];//？？

#pragma omp parallel for  private(lm) num_threads(32)//??
							// 重みの学習を行う
							for (int k = 0; k < alpha_step; k++) {
								if (!is_echo_state_property[k]) continue;     //　https://www.comp.sd.tmu.ac.jp/spacelab/c_lec2/node61.html
								output_learning[k].generate_simultaneous_linear_equationsA(output_node[k][TRAIN], wash_out, step, unit_size);
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
									//std::cout << "成功2" << "\n";
									w[k][lm] = output_learning[k].w;//おそらく論文　手順４　　　　??????????????? [k][lm]→ある入力強み、ユニット間強みの中の、あるλの場合の重み
									//std::cout << "成功3" << "\n";
									nmse[k][lm] = calc_nmse(teacher_signal[VAL], output_learning[k].w, output_node[k][VAL], unit_size, wash_out, step, false);//論文　手順5続き（論文の書き方がややこしくなってるけど、S1(t)を使って求めた重みとS2(t)を使って予測値を計算で求めてNMSEを出そう！！というだけだと思う。）
									//std::cout << "成功4" << "\n";
								}
								
							}
							//std::cout << "成功5" << "\n";
							// 検証データでもっとも性能の良いリザーバーを選択
							for (int k = 0; k < alpha_step; k++) {//論文　手順６
								if (!is_echo_state_property[k]) continue;
								
								
								for (int lm = 0; lm < 10; lm++) {
									//std::cout << "成功6" << "\n";
									if (nmse[k][lm] < opt_nmse) {
										//std::cout << "成功7" << "\n";
										opt_nmse = nmse[k][lm];
										opt_input_signal_factor = k * d_alpha + alpha_min;
										//opt_bias_factor = bias_factor;
										//opt_weight_factor = (k % sigma_step) * d_sigma + sigma_min;
										opt_feed_gain = feed_gain;
										opt_input_gain = input_gain;
										opt_lm2 = lm;
										opt_k = k;
										opt_w = w[k][lm];
										opt_reservoir_layer = reservoir_layer_v[k];
										
										train_nmse = calc_nmse(teacher_signal[TRAIN], opt_w, output_node[opt_k][TRAIN], unit_size, wash_out, step, false);
										//std::cout << "成功8" << "\n";
										//std::cerr << train_nmse << " " << opt_input_signal_factor << " " << opt_feed_gain << " " << opt_input_gain << std::endl;
										
										
										//std::cout << "成功" << "\n";
									}
									//std::cout << "成功" << "\n";

								}
								

							}
							
						}
						//outputfile2 << function_name << "," << opt_input_gain << "," << opt_feed_gain << "," << opt_nmse << std::endl;
					}
					//outputfile2.close();

					/*** TEST phase ***/  //論文　手順7
					std::string output_name = task_name + "_" + std::to_string(param1[r]) + "_" + to_string_with_precision(param2[r], 1) + "_" + function_name + "_" + std::to_string(unit_size) + "_" + std::to_string(loop) + "_" + std::to_string(ite_p);

					std::vector<std::vector<double>> output_node_test(step + 2, std::vector<double>(MAX_NODE_SIZE + 1, 0));// △　+2とか MAX_NODE_SIZEとか
					opt_reservoir_layer.reservoir_update(input_signal[TEST], output_node_test, step);
					//std::cout << "成功9" << "\n";
					test_nmse = calc_nmse(teacher_signal[TEST], opt_w, output_node_test, unit_size, wash_out, step, true, output_name);//l241と引数の数違うけど...
					const double test_nrmse = calc_nrmse(teacher_signal[TEST], opt_w, output_node_test, unit_size, wash_out, step, true, output_name);//l241と引数の数違うけど...

					end = std::chrono::system_clock::now();  // 計測終了時間
					double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); //処理に要した時間をミリ秒に変換

					outputfile << function_name << "," << loop << "," << unit_size << "," << std::fixed << std::setprecision(4) << p << "," << opt_input_signal_factor << "," << opt_input_gain << "," << opt_feed_gain << "," << opt_lm2 << "," << std::fixed << std::setprecision(8) << train_nmse << "," << opt_nmse << "," << test_nmse << "," << test_nrmse << std::endl;
					std::cerr << function_name << "," << loop << "," << unit_size << "," << std::fixed << std::setprecision(3) << p << "," << opt_input_signal_factor << "," << opt_input_gain << "," << opt_feed_gain << "," << opt_lm2 << "," << std::setprecision(5) << train_nmse << "," << opt_nmse << "," << test_nmse << "," << test_nrmse << " " << elapsed / 1000.0 << std::endl;

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






