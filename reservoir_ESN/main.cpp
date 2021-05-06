﻿#include <iostream>
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
double sinc(const double x) {
	if (x == 0) return 1.0;
	return sin(PI * x) / (PI * x);
}
double gauss(double y) { return exp(-y * y / (2.0 * 0.4 * 0.4)) / (sqrt(PI * 2) * 0.4); }
double oddsinc(double y) { 
	if (y <= 0) return sin(PI * y) / (PI * (y + 1));
	else return sin(PI * -y) / (PI * (y - 1));
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
	const int TRIAL_NUM = 1;	// ループ回数
	const int step[3] = { 5000, 2000, 5000 };
	const int wash_out = 100;
	const int task_size = 10;
	std::vector<int> unit_sizes = {300, 300, 300};
	std::vector<std::string> task_names = {  "approx", "approx", "approx"};
	if (unit_sizes.size() != task_names.size()) return 0;
	std::vector<int> param1 = { 7, 3, 5};
	std::vector<double> param2 = { 1.0, 3.0, 1.5 };
	if (param1.size() != param2.size()) return 0;
	std::string task_name;
	std::string function_name;
	const int alpha_step = 21;
	for(int r = 0; r < unit_sizes.size(); r++) {
		const int unit_size = unit_sizes[r];
		const std::string task_name = task_names[r];
		
		std::vector<std::vector<double>> input_signal(PHASE_NUM), teacher_signal(PHASE_NUM);
		
		std::vector<std::string> function_names = { "sinc", "tanh", "gauss", "oddsinc" };
		double alpha_min, d_alpha;
		std::ofstream outputfile("output_data/" + task_name + "_" + std::to_string(param1[r]) + "_" + to_string_with_precision(param2[r], 1) + "_" + std::to_string(unit_size) + ".txt");
		// 入力信号 教師信号の生成
		for (int phase = 0; phase < PHASE_NUM; phase++) {
			
			if (task_name == "narma") {
				d_alpha = 0.01;
				alpha_min = 0.005;
				const int tau = param1[r];
				generate_input_signal_random(input_signal[phase], -1.0, 2.0, step[phase], phase + 1);
				generate_narma_task(input_signal[phase], teacher_signal[phase], tau, step[phase]);
			}
			// 入力分布[-1, 1] -> 出力分布[0, 0.5]のnarmaタスク
			else if (task_name == "narma2") {
				d_alpha = 0.01;
				alpha_min = 0.005;
				const int tau = param1[r];
				generate_input_signal_random(input_signal[phase], -1.0, 2.0, step[phase], phase + 1);
				generate_narma_task2(input_signal[phase], teacher_signal[phase], tau, step[phase]);
			}
			else if (task_name == "henon") {
				d_alpha = 5.0;
				alpha_min = 2.0;
				const int fstep = param1[r];
				generate_henom_map_task(input_signal[phase], teacher_signal[phase], fstep, step[phase], phase * step[TRAIN]);
			}
			else if (task_name == "laser") {
				d_alpha = 0.2;
				alpha_min = 0.1;
				const int fstep = param1[r];
				generate_laser_task(input_signal[phase], teacher_signal[phase], fstep, step[phase], phase * step[phase]);
			}
			else if (task_name == "approx") {
				d_alpha = 1.0;
				alpha_min = 1.0;
				const int tau = param1[r];
				const double nu = param2[r];
				
				generate_input_signal_random(input_signal[phase], -1.0, 2.0, step[phase], phase + 1);
				task_for_function_approximation(input_signal[phase], teacher_signal[phase], nu, tau, step[phase], phase);
			}
		}

		// 設定出力
		outputfile << "### task_name: " << task_name << std::endl;
		outputfile << "### " << param1[r] << " " << param2[r] << std::endl;
		outputfile << "### input_signal_factor [" << alpha_min << ", " << alpha_min + d_alpha * 10 << "]" << std::endl;
		outputfile << "### weight_factor [0.1, 1.1]" << std::endl;
		outputfile << "function_name,seed,unit_size,p,input_singal_factor,weight_factor,lm,train_nmse,nmse,test_nmse" << std::endl;

		std::chrono::system_clock::time_point  start, end; // 型は auto で可
		for (auto function_name : function_names) {
			double (*nonlinear)(double);
			if (function_name == "sinc") nonlinear = sinc;
			else if (function_name == "tanh") nonlinear = tanh;
			else if (function_name == "gauss") nonlinear = gauss;
			else if (function_name == "oddsinc") nonlinear = oddsinc;
			for (int loop = 0; loop < TRIAL_NUM; loop++) {
				std::vector<std::vector<std::vector<double>>> output_node_TRAIN(alpha_step * 11, std::vector<std::vector<double>>(step[TRAIN] + 2, std::vector<double>(unit_size + 1, 0)));
				std::vector<std::vector<std::vector<double>>> output_node_VAL(alpha_step * 11, std::vector<std::vector<double>>(step[VAL] + 2, std::vector<double>(unit_size + 1, 0)));
				std::vector<std::vector<double>> output_node_TEST(step[TEST] + 2, std::vector<double>(unit_size + 1, 0));
				std::vector<reservoir_layer> reservoir_layer_v(alpha_step * 11);
				std::vector<bool> is_echo_state_property(alpha_step * 11);
				for (int ite_p = 0; ite_p <= 10; ite_p += 1) {
					double opt_nmse = 1e+10;
					double opt_input_signal_factor = 0;
					double opt_weight_factor = 0;
					double opt_lm2 = 0;
					double test_nmse = 1e+10;
					start = std::chrono::system_clock::now(); // 計測開始時間

#pragma omp parallel for
					// 複数のリザーバーの時間発展をまとめて処理
					for (int k = 0; k < alpha_step * 11; k++) {
						
						const double p = ite_p * 0.1;
						const double input_signal_factor = (k / alpha_step) * d_alpha + alpha_min;
						const double weight_factor = (k % 11 + 1) * 0.1;

						reservoir_layer reservoir_layer1(unit_size, unit_size / 10, input_signal_factor, weight_factor, p, nonlinear, loop, wash_out);
						reservoir_layer1.generate_reservoir();

						reservoir_layer1.reservoir_update(input_signal[TRAIN], output_node_TRAIN[k], step[TRAIN]);
						reservoir_layer1.reservoir_update(input_signal[VAL], output_node_VAL[k], step[VAL]);
						is_echo_state_property[k] = reservoir_layer1.is_echo_state_property(input_signal[VAL]);
						reservoir_layer_v[k] = reservoir_layer1;
					}
					int lm;
					std::vector<std::vector<std::vector<double>>> w(alpha_step * 11, std::vector<std::vector<double>>(10)); // 各リザーバーの出力重み
					std::vector<std::vector<double>> nmse(alpha_step * 11, std::vector<double>(10));						// 各リザーバーのnmseを格納
					int opt_k = 0;
					//#pragma omp parallel for
					// 重みの学習を行う
					for (int k = 0; k < alpha_step * 11; k++) {
						if (!is_echo_state_property[k]) continue;
						output_learning output_learning;
						const double p = ite_p * 0.1;
						const double input_signal_factor = (k / 11) * d_alpha + alpha_min;
						const double weight_factor = (k % 11) * 0.1;
						output_learning.generate_simultaneous_linear_equationsA(output_node_TRAIN[k], wash_out, step[TRAIN], unit_size);
						output_learning.generate_simultaneous_linear_equationsb(output_node_TRAIN[k], teacher_signal[TRAIN], wash_out, step[TRAIN], unit_size);

						double opt_lm = 0;
						double opt_lm_nmse = 1e+9;
						for (lm = 0; lm < 10; lm++) {
							for (int j = 0; j <= unit_size; j++) {
								output_learning.A[j][j] += pow(10, -12 + lm/2.0);
								if (lm != 0) output_learning.A[j][j] -= pow(10, -12 + lm/2.0 - 1);
							}
							output_learning.IncompleteCholeskyDecomp2(unit_size + 1);
							double eps = 1e-16;
							int itr = 300;
							output_learning.ICCGSolver(unit_size + 1, itr, eps);
							w[k][lm] = output_learning.w;
							nmse[k][lm] = calc_nmse(teacher_signal[VAL], output_learning.w, output_node_VAL[k], unit_size, wash_out, step[VAL], false);
						}
					}
					std::vector<double> opt_w;
					// 検証データでもっとも性能の良いリザーバーを選択
					for (int k = 0; k < alpha_step * 11; k++) {
						if (!is_echo_state_property[k]) continue;
						for (int lm = 0; lm < 10; lm++) {
							if (nmse[k][lm] < opt_nmse) {
								opt_nmse = nmse[k][lm];
								opt_input_signal_factor = (k / 11) * d_alpha + alpha_min;
								opt_weight_factor = (k % 11 + 1) * 0.1;
								opt_lm2 = lm;
								opt_k = k;
								opt_w = w[k][lm];
							}
						}

					}
					/*** TEST phase ***/
					std::string output_name = task_name + "_" + std::to_string(param1[r]) + "_" + to_string_with_precision(param2[r], 1) + "_" + function_name + "_" + std::to_string(unit_size) + "_" + std::to_string(loop) + "_" + std::to_string(ite_p);
		
					reservoir_layer_v[opt_k].reservoir_update(input_signal[TEST], output_node_TEST, step[TEST]);
					test_nmse = calc_nmse(teacher_signal[TEST], opt_w, output_node_TEST, unit_size, wash_out, step[TEST], true, output_name);
					double train_nmse = calc_nmse(teacher_signal[TRAIN], opt_w, output_node_TRAIN[opt_k], unit_size, wash_out, step[TRAIN], true, output_name);
					end = std::chrono::system_clock::now();  // 計測終了時間
					double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); //処理に要した時間をミリ秒に変換
					
					outputfile << function_name << "," << loop << "," << unit_size << "," << std::fixed << std::setprecision(2) << ite_p * 0.1 << "," << opt_input_signal_factor << "," << opt_weight_factor << "," << opt_lm2 << "," << std::fixed << std::setprecision(8) << train_nmse <<"," <<opt_nmse << "," << test_nmse << std::endl;
					std::cerr << function_name << "," << loop << "," << unit_size << "," << std::fixed << std::setprecision(2) << ite_p * 0.1 << "," << opt_input_signal_factor << "," << opt_weight_factor << "," << opt_lm2 << "," << std::fixed << std::setprecision(4) << train_nmse <<"," <<opt_nmse << "," << test_nmse << " " << elapsed / 1000.0 << std::endl;
					for (int k = 0; k < alpha_step * 11; k++) {
						if (!is_echo_state_property[k]) continue;
						for (int lm = 0; lm < 10; lm++) {
							std::cerr << k << " " << lm << " " << nmse[k][lm] << std::endl;
						}

					}
					for(int i = 0; test_nmse/opt_nmse > 1.5 && i < 10; i++) {
						
						input_signal[TEST].clear();
						teacher_signal[TEST].clear();
						if (task_name == "narma") {
							const int tau = param1[r];
							generate_input_signal_random(input_signal[TEST], -1.0, 2.0, step[TEST], TEST + i);
							generate_narma_task(input_signal[TEST], teacher_signal[TEST], tau, step[TEST]);
						}
						else if (task_name == "narma2") {
							const int tau = param1[r];
							generate_input_signal_random(input_signal[TEST], -1.0, 2.0, step[TEST], TEST + i);
							generate_narma_task2(input_signal[TEST], teacher_signal[TEST], tau, step[TEST]);
						}
						else if (task_name == "henon") {
							break;
						}
						else if (task_name == "laser") {
							break;
						}
						else if (task_name == "approx") {
							const int tau = param1[r];
							const double nu = param2[r];
							generate_input_signal_random(input_signal[TEST], -1.0, 2.0, step[TEST], TEST + i);
							task_for_function_approximation(input_signal[TEST], teacher_signal[TEST], nu, tau, step[TEST], TEST + i);
						}
						reservoir_layer_v[opt_k].reservoir_update(input_signal[TEST], output_node_TEST, step[TEST]);
						test_nmse = calc_nmse(teacher_signal[TEST], opt_w, output_node_TEST, unit_size, wash_out, step[TEST], true, output_name);
						opt_nmse = calc_nmse(teacher_signal[VAL], opt_w, output_node_VAL[opt_k], unit_size, wash_out, step[VAL], false);
						double train_nmse = calc_nmse(teacher_signal[TRAIN], opt_w, output_node_TRAIN[opt_k], unit_size, wash_out, step[TEST], true, output_name);
						end = std::chrono::system_clock::now();  // 計測終了時間
						double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); //処理に要した時間をミリ秒に変換

						outputfile << function_name << "," << loop << "," << unit_size << "," << std::fixed << std::setprecision(2) << ite_p * 0.1 << "," << opt_input_signal_factor << "," << opt_weight_factor << "," << opt_lm2 << "," << std::fixed << std::setprecision(8) << train_nmse << "," << opt_nmse << "," << test_nmse << std::endl;
						std::cerr << function_name << "," << loop << "," << unit_size << "," << std::fixed << std::setprecision(2) << ite_p * 0.1 << "," << opt_input_signal_factor << "," << opt_weight_factor << "," << opt_lm2 << "," << std::fixed << std::setprecision(4) << train_nmse << "," << opt_nmse << "," << test_nmse << " " << elapsed / 1000.0 << std::endl;
						
					}
					// リザーバーのユニット入出力を表示
					reservoir_layer_v[opt_k].reservoir_update_show(input_signal[TEST], output_node_TEST, step[TEST], wash_out, output_name);

				}

			}
		}
		outputfile.close();
	}
}
