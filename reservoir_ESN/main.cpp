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
	const int step = 3000;
	const int wash_out = 500;
	const int task_size = 10;
	std::vector<int> unit_sizes = {100, 100, 100, 100, 100, 100, 100, 100, 100, 100,  100, 100, 100, 100 };
	std::vector<std::string> task_names = { "laser", "laser", "henon", "henon", "narma", "narma", "narma", "narma2", "narma2", "narma2", "approx", "approx", "approx"};
	if (unit_sizes.size() != task_names.size()) return 0;
	std::vector<int> param1 = { 5, 7, 5, 7, 5, 10, 15, 5, 10, 15, 3, 5, 7 };
	std::vector<double> param2 = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.0, 1.5, 1.0 };
	if (param1.size() != param2.size()) return 0;
	std::string task_name;
	std::string function_name;

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
				generate_input_signal_random(input_signal[phase], -1.0, 2.0, step, phase + 1);
				generate_narma_task(input_signal[phase], teacher_signal[phase], tau, step);
			}
			// 入力分布[-1, 1] -> 出力分布[0, 0.5]のnarmaタスク
			else if (task_name == "narma2") {
				d_alpha = 0.01;
				alpha_min = 0.005;
				const int tau = param1[r];
				generate_input_signal_random(input_signal[phase], -1.0, 2.0, step, phase + 1);
				generate_narma_task2(input_signal[phase], teacher_signal[phase], tau, step);
			}
			else if (task_name == "henon") {
				d_alpha = 5.0;
				alpha_min = 2.0;
				const int fstep = param1[r];
				generate_henom_map_task(input_signal[phase], teacher_signal[phase], fstep, step, phase * step);
			}
			else if (task_name == "laser") {
				d_alpha = 0.2;
				alpha_min = 0.1;
				const int fstep = param1[r];
				generate_laser_task(input_signal[phase], teacher_signal[phase], fstep, step, phase * step);
			}
			else if (task_name == "approx") {
				d_alpha = 2.0;
				alpha_min = 1.0;
				const int tau = param1[r];
				const double nu = param2[r];
				
				generate_input_signal_random(input_signal[phase], -1.0, 2.0, step, phase + 1);
				task_for_function_approximation(input_signal[phase], teacher_signal[phase], nu, tau, step, phase);
			}
		}

		// 設定出力
		outputfile << "### task_name: " << task_name << std::endl;
		outputfile << "### " << param1[r] << " " << param2[r] << std::endl;
		outputfile << "### input_signal_factor [" << alpha_min << ", " << alpha_min + d_alpha * 10 << "]" << std::endl;
		outputfile << "### weight_factor [0.1, 1.1]" << std::endl;
		outputfile << "function_name,seed,unit_size,p,input_singal_factor,weight_factor,lm,nmse,test_nmse" << std::endl;

		std::chrono::system_clock::time_point  start, end; // 型は auto で可
		for (auto function_name : function_names) {
			double (*nonlinear)(double);
			if (function_name == "sinc") nonlinear = sinc;
			else if (function_name == "tanh") nonlinear = tanh;
			else if (function_name == "gauss") nonlinear = gauss;
			else if (function_name == "oddsinc") nonlinear = oddsinc;
			for (int loop = 0; loop < TRIAL_NUM; loop++) {
				std::vector<std::vector<std::vector<std::vector<double>>>> output_node(21 * 11, std::vector<std::vector<std::vector<double>>>(PHASE_NUM, std::vector<std::vector<double>>(step + 2, std::vector<double>(unit_size + 1, 0))));
				std::vector<reservoir_layer> reservoir_layer_v(21 * 11);
				for (int ite_p = 0; ite_p <= 10; ite_p += 1) {
					double opt_nmse = 1e+10;
					double opt_input_signal_factor = 0;
					double opt_weight_factor = 0;
					double opt_lm2 = 0;
					double test_nmse = 1e+10;
					start = std::chrono::system_clock::now(); // 計測開始時間

#pragma omp parallel for
					// 複数のリザーバーの時間発展をまとめて処理
					for (int k = 0; k < 21 * 11; k++) {
						const double p = ite_p * 0.1;
						const double input_signal_factor = (k / 11) * d_alpha + alpha_min;
						const double weight_factor = (k % 11 + 1) * 0.1;

						reservoir_layer reservoir_layer1(unit_size, unit_size / 10, input_signal_factor, weight_factor, p, nonlinear, loop, wash_out);
						reservoir_layer1.generate_reservoir();

						reservoir_layer1.reservoir_update(input_signal[TRAIN], output_node[k][TRAIN], step);
						reservoir_layer1.reservoir_update(input_signal[VAL], output_node[k][VAL], step);
						reservoir_layer_v[k] = reservoir_layer1;
					}
					int lm;
					std::vector<std::vector<std::vector<double>>> w(21 * 11, std::vector<std::vector<double>>(10)); // 各リザーバーの出力重み
					std::vector<std::vector<double>> nmse(21 * 11, std::vector<double>(10));						// 各リザーバーのnmseを格納
					int opt_k = 0;
					//#pragma omp parallel for
					// 重みの学習を行う
					for (int k = 0; k < 21 * 11; k++) {
						output_learning output_learning;
						const double p = ite_p * 0.1;
						const double input_signal_factor = (k / 11) * d_alpha + alpha_min;
						const double weight_factor = (k % 11) * 0.1;
						output_learning.generate_simultaneous_linear_equationsA(output_node[k][TRAIN], wash_out, step, unit_size);
						output_learning.generate_simultaneous_linear_equationsb(output_node[k][TRAIN], teacher_signal[TRAIN], wash_out, step, unit_size);

						double opt_lm = 0;
						double opt_lm_nmse = 1e+9;
						for (lm = 0; lm < 10; lm++) {
							for (int j = 0; j <= unit_size; j++) {
								output_learning.A[j][j] += pow(10, -16 + lm);
								if (lm != 0) output_learning.A[j][j] -= pow(10, -16 + lm - 1);
							}
							output_learning.IncompleteCholeskyDecomp2(unit_size + 1);
							double eps = 1e-12;
							int itr = 10;
							output_learning.ICCGSolver(unit_size + 1, itr, eps);
							w[k][lm] = output_learning.w;
							nmse[k][lm] = calc_nmse(teacher_signal[VAL], output_learning.w, output_node[k][VAL], unit_size, wash_out, step, false);
						}
					}
					std::vector<double> opt_w;
					// 検証データでもっとも性能の良いリザーバーを選択
					for (int k = 0; k < 21 * 11; k++) {
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

					reservoir_layer_v[opt_k].reservoir_update(input_signal[TEST], output_node[opt_k][TEST], step);
					test_nmse = calc_nmse(teacher_signal[TEST], opt_w, output_node[opt_k][TEST], unit_size, wash_out, step, true, output_name);
					end = std::chrono::system_clock::now();  // 計測終了時間
					double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); //処理に要した時間をミリ秒に変換
					
					outputfile << function_name << "," << loop << "," << unit_size << "," << std::fixed << std::setprecision(1) << ite_p * 0.1 << "," << opt_input_signal_factor << "," << opt_weight_factor << "," << opt_lm2 << "," << std::fixed << std::setprecision(8) << opt_nmse << "," << test_nmse << std::endl;
					std::cerr << function_name << "," << loop << "," << unit_size << "," << std::fixed << std::setprecision(1) << ite_p * 0.1 << "," << opt_input_signal_factor << "," << opt_weight_factor << "," << opt_lm2 << "," << std::fixed << std::setprecision(4) << opt_nmse << "," << test_nmse << " " << elapsed / 1000.0 << std::endl;

					// リザーバーのユニット入出力を表示
					reservoir_layer_v[opt_k].reservoir_update_show(input_signal[TEST], output_node[opt_k][TEST], step, wash_out, output_name);

				}

			}
		}
		outputfile.close();
	}
}
