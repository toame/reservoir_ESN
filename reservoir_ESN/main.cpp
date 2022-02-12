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
#define TRAIN (0)
#define VAL (1)
#define TEST (2)
#define MAX_NODE_SIZE (500)
double TD_MG(const double x, double J, double input_gain, double feed_gain) {
	return (feed_gain * (x + input_gain * J)) / (1.0 + pow(x + input_gain * J, 4.0));//ρ = 1～10  //////////////変更要素/////////////////
}

double TD_ikeda(const double x, double J, double input_gain, double feed_gain) {
	return feed_gain * pow(sin(x + input_gain * J + 0.5), 2.0);//Φ=0.2～0.7(0.05刻み)  //////////////変更要素/////////////////
}

double TD_exp(const double x, double J, double input_gain, double feed_gain) {
	return feed_gain * exp(-x) * sin(x + input_gain * J);
}


#include <sstream>

template <typename T>

std::string to_string_with_precision(const T a_value, const int n = 6)
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
	std::vector<int> unit_sizes = { 20 };///////////////////変更要素////////////////////////

	std::vector<std::string> task_names = { "henon"};///////////////////変更要素////////////////////////
	if (unit_sizes.size() != task_names.size()) return 0;
	std::vector<int> param1 = { 5 };///////////////////変更要素////////////////////////
	std::vector<double> param2 = {0.0};///////////////////変更要素////////////////////////
	if (param1.size() != param2.size()) return 0;
	const int alpha_step = 11;
	const int sigma_step = 11;
	std::string task_name;
	std::string function_name;
	std::vector<std::vector<std::vector<std::vector<double>>>> output_node(alpha_step * sigma_step, std::vector<std::vector<std::vector<double>>>(PHASE_NUM, std::vector<std::vector<double>>(step + 2, std::vector<double>(MAX_NODE_SIZE + 1, 0))));
	std::vector<reservoir_layer> reservoir_layer_v(alpha_step * sigma_step);
	std::vector<bool> is_echo_state_property(alpha_step * sigma_step);
	std::vector<std::vector<std::vector<double>>> w(alpha_step/* * sigma_step, */,std::vector<std::vector<double>>(10)); 
	std::vector<std::vector<double>> nmse(alpha_step * sigma_step, std::vector<double>(10));	
	for (int r = 0; r < unit_sizes.size(); r++) {
		const int unit_size = unit_sizes[r];
		const std::string task_name = task_names[r];
		std::vector<std::vector<double>> input_signal(PHASE_NUM), teacher_signal(PHASE_NUM);

		std::vector<std::string> function_names = { "TD_MG",  "TD_ikeda", };// "STDE_MG", "STDE_ikeda", "TD_exp"  ///////////////////変更要素////////////////////////
		double alpha_min, d_alpha;
		double sigma_min, d_sigma;
		double d_bias;
		std::ofstream outputfile("output_data_STDE/" + task_name + "_" + std::to_string(param1[r]) + "_" + to_string_with_precision(param2[r], 1) + "_" + std::to_string(unit_size) + ".txt");
		// 入力信号 教師信号の生成
		for (int phase = 0; phase < PHASE_NUM; phase++) {
			
			if (task_name == "narma") {
				d_alpha = 0.01; alpha_min = 0.1;
				const int tau = param1[r];
				generate_input_signal_random(input_signal[phase], -1.0, 2.0, step, phase + 1);
				generate_narma_task(input_signal[phase], teacher_signal[phase], tau, step);
			}
			// 入力分布[-1, 1] -> 出力分布[0, 0.5]のnarmaタスク
			else if (task_name == "narma2") {
				d_alpha = 0.005; alpha_min = 0.002;
				const int tau = param1[r];
				generate_input_signal_random(input_signal[phase], -1.0, 2.0, step, phase + 1);
				generate_narma_task2(input_signal[phase], teacher_signal[phase], tau, step);
			}
			else if (task_name == "henon") {
				d_alpha = 0.2; alpha_min = 0.0;
				const int fstep = param1[r];
				generate_henom_map_task(input_signal[phase], teacher_signal[phase], fstep, step, phase * step);
			}
			else if (task_name == "henon2") {
				d_alpha = 0.2; alpha_min = 0.0;
				const int fstep = param1[r];
				generate_henom_map_task2(input_signal[phase], teacher_signal[phase], fstep, step, phase * step);
			}
			else if (task_name == "laser") {
				d_alpha = 2.0; alpha_min = 0.1;
				const int fstep = param1[r];
				generate_laser_task(input_signal[phase], teacher_signal[phase], fstep, step, phase * step);
			}
			else if (task_name == "approx") {
				const int tau = param1[r];
				const double nu = param2[r];
				/*if (tau == 7) { d_alpha = 1.0; alpha_min = 0.1;}
				else if (tau == 5) { d_alpha = 2.0; alpha_min = 0.5;}
				else if (tau == 3) { d_alpha = 5.0; alpha_min = 1.0;}
				else if (tau == 1) {
					d_alpha = 10.0; alpha_min = 1.0;
				}
				else {
					std::cerr << "error! approx parameter is not setting" << std::endl;
					return 0;
				}*/

				generate_input_signal_random(input_signal[phase], -1.0, 2.0, step, phase + 1);
				task_for_function_approximation(input_signal[phase], teacher_signal[phase], nu, tau, step, phase);
			}
			else if (task_name == "approx2") {
				const int tau = param1[r];
				const double nu = param2[r];
				//if (tau == 7) { d_alpha = 1.0; alpha_min = 0.1;}
				//else if (tau == 5) { d_alpha = 2.0; alpha_min = 0.5;}
				//else if (tau == 3 || tau == 2) { 
				//	//d_alpha = 1.0; alpha_min = 1.0;
				//}
				//else if (tau == 1) {
				//	d_alpha = 10.0; alpha_min = 1.0;
				//}
				//else {
				//	std::cerr << "error! approx parameter is not setting" << std::endl;
				//	return 0;
				//}

				generate_input_signal_random(input_signal[phase], -1.0, 2.0, step, phase + 1);
				task_for_function_approximation2(input_signal[phase], teacher_signal[phase], nu, tau, step, phase);
			}
			/*
			else if (task_name == "legendre") {
				const int tau = param1[r];
				const double nu = param2[r];
				if (tau == 7) { d_alpha = 1.0; alpha_min = 0.1;}
				else if (tau == 5) { d_alpha = 2.0; alpha_min = 0.5;}
				else if (tau == 3) { d_alpha = 5.0; alpha_min = 1.0;}
				else if (tau == 0) {
					d_alpha = 30.0; alpha_min = 100.0;
				}
				else {
					std::cerr << "error! legendre parameter is not setting" << std::endl;
					return 0;
				}

				generate_input_signal_random(input_signal[phase], -1.0, 2.0, step, phase + 1);
				generate_legendre_task(input_signal[phase], teacher_signal[phase], nu, tau, step);
			}*/
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
			double (*nonlinear)(double, double, double, double);
			if (function_name == "TD_MG") {
				nonlinear = TD_MG;
				d_alpha = 0.2; alpha_min = 0.0;///////////////////変更要素////////////////////////
			}
			else if (function_name == "TD_ikeda") {
				nonlinear = TD_ikeda;
				d_alpha = 2.0; alpha_min = 0.0;///////////////////変更要素////////////////////////
			}
			else if (function_name == "TD_exp") {
				nonlinear = TD_exp;
				d_alpha = 0.05; alpha_min = 0.1;///////////////////変更要素////////////////////////
			}
			else {
				std::cerr << "error! " << function_name << "is not found" << std::endl;
				return 0;
			}
			for (int loop = 0; loop < TRIAL_NUM; loop++) {
				for (int ite_p = 0; ite_p <= 10; ite_p += 1) {
					const double p = ite_p * 0.1;
					double opt_nmse = 1e+10;//opt 最適な値  
					double opt_input_signal_factor = 0;
					//double opt_bias_factor = 0;
					//double opt_weight_factor = 0;
					double opt_feed_gain = 0;//最適なフィードバックゲイン
					double opt_input_gain = 0;//最適な入力ゲイン
					double opt_lm2 = 0;//lmはλ
					double test_nmse = 1e+10;
					double train_nmse = 1e+10;
					reservoir_layer opt_reservoir_layer;
					std::vector<double> opt_w;
					start = std::chrono::system_clock::now(); // 計測開始時間
					for (int ite_input = 1; ite_input <= 10; ite_input += 1) {
						const double input_gain = 0.7 + ite_input * 0.05;///////////////////変更要素////////////////////////
						for (int ite_feed = 1; ite_feed <= 10; ite_feed += 1) {
							const double feed_gain = 0.10 + ite_feed * 0.02;//////////////////////変更要素/////////////////////////////
#pragma omp parallel for num_threads(32)
						// 複数のリザーバーの時間発展をまとめて処理
							for (int k = 0; k < alpha_step; k++) {
								const double input_signal_factor = k * d_alpha + alpha_min;
								reservoir_layer reservoir_layer1(unit_size, input_signal_factor, input_gain, feed_gain, p, nonlinear, loop, wash_out, step);
								reservoir_layer1.generate_reservoir();
								reservoir_layer1.reservoir_update(input_signal[TRAIN], output_node[k][TRAIN], step);
								reservoir_layer1.reservoir_update(input_signal[VAL], output_node[k][VAL], step);
								is_echo_state_property[k] = reservoir_layer1.is_echo_state_property(input_signal[VAL]);
								reservoir_layer_v[k] = reservoir_layer1;
							}
							int lm;

							int opt_k = 0;

							output_learning output_learning[341];

#pragma omp parallel for  private(lm) num_threads(32)
							// 重みの学習を行う
							for (int k = 0; k < alpha_step; k++) {
								if (!is_echo_state_property[k]) continue;     
								output_learning[k].generate_simultaneous_linear_equationsA(output_node[k][TRAIN], wash_out, step, unit_size);
								output_learning[k].generate_simultaneous_linear_equationsb(output_node[k][TRAIN], teacher_signal[TRAIN], wash_out, step, unit_size);

								double opt_lm = 0;
								double opt_lm_nmse = 1e+9;
								
								for (lm = 0; lm < 10; lm++) {
									for (int j = 0; j <= unit_size; j++) {
										output_learning[k].A[j][j] += pow(10, -15 + lm);
										if (lm != 0) output_learning[k].A[j][j] -= pow(10, -16 + lm);
									}
									output_learning[k].IncompleteCholeskyDecomp2(unit_size + 1);
									double eps = 1e-12;
									int itr = 10;
									output_learning[k].ICCGSolver(unit_size + 1, itr, eps);
									w[k][lm] = output_learning[k].w;
									nmse[k][lm] = calc_nmse(teacher_signal[VAL], output_learning[k].w, output_node[k][VAL], unit_size, wash_out, step, false);
								}
							}
							// 検証データでもっとも性能の良いリザーバーを選択
							for (int k = 0; k < alpha_step; k++) {//論文　手順６
								if (!is_echo_state_property[k]) continue;
								for (int lm = 0; lm < 10; lm++) {
									if (nmse[k][lm] < opt_nmse) {
										opt_nmse = nmse[k][lm];
										opt_input_signal_factor = k * d_alpha + alpha_min;
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
						}
					}
					/*** TEST phase ***/  
					std::string output_name = task_name + "_" + std::to_string(param1[r]) + "_" + to_string_with_precision(param2[r], 1) + "_" + function_name + "_" + std::to_string(unit_size) + "_" + std::to_string(loop) + "_" + std::to_string(ite_p);

					std::vector<std::vector<double>> output_node_test(step + 2, std::vector<double>(MAX_NODE_SIZE + 1, 0));
					opt_reservoir_layer.reservoir_update(input_signal[TEST], output_node_test, step);
					test_nmse = calc_nmse(teacher_signal[TEST], opt_w, output_node_test, unit_size, wash_out, step, true, output_name);
					const double test_nrmse = calc_nrmse(teacher_signal[TEST], opt_w, output_node_test, unit_size, wash_out, step, true, output_name);

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