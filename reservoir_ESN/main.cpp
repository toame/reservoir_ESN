#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <cblas.h>
#include <chrono>
#include <random>
#include <algorithm>
#include <cstring>
#include <execution>
#include <memory>
#include <numeric>
#include "reservoir_layer.h"
#include "output_learning.h"
#include "task.h"
#define PHASE_NUM (3)
#define TRAIN (0)
#define VAL (1)
#define TEST (2)
#define MAX_UNIT_SIZE (500)
#define MAX_TASK_SIZE (2000)
#define TRUNC_EPSILON (1.7e-4)
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

void unit_test() {
	const int M = 100, N = 3000;
	std::vector<double> n(M, 0), n2(M, 0), y(N);
	std::vector<std::vector<double>> r(N, std::vector<double>(M));
	std::vector<std::vector<double>> r2(M, std::vector<double>(N));
	std::vector<double> r3;

	for (int t = 0; t < N; t++) y[t] = (t * 3 + 2) % 10 / 3.0 - 0.5;
	for (int i = 0; i < M; i++) {
		for (int t = 0; t < N; t++) {
			r[t][i] = (t * 3 + 2 + i * 5) % 31 / 100.0 - 0.01;
			r2[i][t] = (t * 3 + 2 + i * 5) % 31 / 100.0 - 0.01;
		}
	}
	for (int i = 0; i < M * N; i++) {
		r3.push_back(r[i % N][i / N]);
	}
	for (int i = 0; i < M; i++) {
		for (int t = 0; t < N; t++) {
			n[i] += r2[i][t] * y[t];
		}
	}
	double alpha = 1.0, beta = 0.0;
	cblas_dgemv(CblasRowMajor, CblasNoTrans, M, N, alpha, r3.data(), N, y.data(), 1, beta, n2.data(), 1);
	for (int i = 0; i < M; i++) {
		std::cerr << i << " " << n[i] << " " << n2[i] << std::endl;
	}
}

int main(void) {

	const int TRIAL_NUM = 5;	// ループ回数
	const int step = 3000;
	const int wash_out = 500;
	std::vector<int> unit_sizes = { 100 };
	std::vector<std::string> task_names = { "NL" };
	if (unit_sizes.size() != task_names.size()) return 0;
	std::vector<int> param1 = {
								   0 };
	std::vector<double> param2 = {
									0 };
	if (param1.size() != param2.size()) return 0;
	std::vector<double> bias_set{ 0, 1, 2, 3, 5, 10, 20, 40 };
	std::vector<double> alpha_set{ 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0, 50.0, 70.0 };
	std::vector<double> sigma_set{ 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2 };
	const int alpha_step = alpha_set.size();
	const int sigma_step = sigma_set.size();
	const int lambda_step = 5;
	std::string task_name;
	std::string function_name;

	std::vector<std::vector<std::vector<std::vector<double>>>> output_node(alpha_step * sigma_step, std::vector<std::vector<std::vector<double>>>(PHASE_NUM, std::vector<std::vector<double>>(step + 2, std::vector<double>(MAX_UNIT_SIZE + 1, 0))));
	std::vector<reservoir_layer> reservoir_layer_v(alpha_step * sigma_step);
	std::vector<bool> is_echo_state_property(alpha_step * sigma_step);
	std::vector < std::vector<std::vector<std::vector<double>>>> w(alpha_step * sigma_step, std::vector<std::vector<std::vector<double>>>(MAX_TASK_SIZE, std::vector<std::vector<double>>(lambda_step))); // 各リザーバーの出力重み
	std::vector<std::vector<std::vector<double>>> nmse(alpha_step * sigma_step, std::vector<std::vector<double>>(MAX_TASK_SIZE, std::vector<double>(lambda_step)));						// 各リザーバーのnmseを格納
	std::vector<double> approx_nu_set({ 0.5, 0.7, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0 });
	std::vector<int> approx_tau_set({ 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20 });
	std::vector<int> narma_tau_set({ 5, 7, 8, 10, 12, 13, 15, 17, 18, 20 });
	for (int r = 0; r < unit_sizes.size(); r++) {
		const int unit_size = unit_sizes[r];
		const std::string task_name = task_names[r];

		std::vector<std::vector<double>> input_signal(PHASE_NUM);
		std::vector<std::vector<std::vector<double>>> teacher_signals(PHASE_NUM);

		std::vector<std::string> function_names = { "sinc", "tanh"};
		double alpha_min, d_alpha;
		double sigma_min, d_sigma;
		
		std::ofstream outputfile("output_data/" + task_name + "_" + std::to_string(param1[r]) + "_" + to_string_with_precision(param2[r], 1) + "_" + std::to_string(unit_size) + ".csv");
		// 入力信号 教師信号の生成
		std::vector<std::vector<std::vector<int>>> d_vec(PHASE_NUM);
		std::vector<std::string> task_name2;
		std::vector<int> task_size(4);
		generate_d_sequence_set(d_vec);
		
		for (int phase = 0; phase < PHASE_NUM; phase++) {
			generate_input_signal_random(input_signal[phase], -1.0, 2.0, step, phase + 1);
			for (auto tau : narma_tau_set) {
				std::vector<double> tmp_teacher_signal;
				generate_narma_task2(input_signal[phase], tmp_teacher_signal, tau, step);
				teacher_signals[phase].push_back(tmp_teacher_signal);
				if(phase == TRAIN) task_name2.push_back("narma" + std::to_string(tau));
			}
			task_size[0] = teacher_signals[phase].size();
			for (auto nu: approx_nu_set) {
				for (auto tau : approx_tau_set) {
					std::vector<double> tmp_teacher_signal;
					task_for_function_approximation(input_signal[phase], tmp_teacher_signal, nu, tau, step, phase);
					teacher_signals[phase].push_back(tmp_teacher_signal);
					if (phase == TRAIN) task_name2.push_back("approx" + std::to_string(tau) + "_" + to_string_with_precision(nu, 1));
				}
			}
			task_size[1] = teacher_signals[phase].size();
			for (int tau = 1; tau <= unit_size; tau++) {
				std::vector<double> tmp_teacher_signal;
				task_for_calc_of_L(input_signal[phase], tmp_teacher_signal, tau, step);
				teacher_signals[phase].push_back(tmp_teacher_signal);
			}
			task_size[2] = teacher_signals[phase].size();
			std::cerr << "ok: " << d_vec[phase].size() << std::endl;
			for (auto& d : d_vec[phase]) {
				std::vector<double> tmp_teacher_signal;
				task_for_calc_of_NL(input_signal[phase], tmp_teacher_signal, d, step);
				teacher_signals[phase].push_back(tmp_teacher_signal);
			}
			task_size[3] = teacher_signals[phase].size();
		}
		// 設定出力
		outputfile << "### task_name: " << task_name << std::endl;
		outputfile << "### " << param1[r] << " " << param2[r] << std::endl;
		outputfile << "### input_signal_factor [" << alpha_set.front() << ", " << alpha_set.back() << "]" << std::endl;
		outputfile << "### weight_factor [" << sigma_set.front() << ", " << sigma_set.back() << "]" << std::endl;
		outputfile << "### bias_factor [" << bias_set.front() << ", " << bias_set.back() << "]" << std::endl;
		outputfile << "function_name,seed,unit_size,p,input_singal_factor,bias_factor,weight_factor,train_L,L,train_NL,NL,NL_1";
		for (int i = 2; i < 30; i++) {
			outputfile << ",NL" << std::to_string(i);
		}
		for (int i = 0; i < task_name2.size(); i++) {
			outputfile << "," << task_name2[i];
		}
		outputfile << std::endl;
		std::cerr << teacher_signals[TRAIN].size() << std::endl;
		for (int i = 0; i < 4; i++) std::cerr << task_size[i] << " ";
		std::cerr << std::endl;
		std::chrono::system_clock::time_point  start, end; // 型は auto で可
		for (int loop = 0; loop < TRIAL_NUM; loop++) {
			for (auto function_name : function_names) {
				double (*nonlinear)(double);
				if (function_name == "sinc") nonlinear = sinc;
				else if (function_name == "tanh") nonlinear = tanh;
				else if (function_name == "gauss") nonlinear = gauss;
				else if (function_name == "oddsinc") nonlinear = oddsinc;
				else {
					std::cerr << "error! " << function_name << "is not found" << std::endl;
					return 0;
				}
				for (int ite_p = 0; ite_p < 10; ite_p += 1) {
					const double p = ite_p * 0.1 + 0.1;
					start = std::chrono::system_clock::now(); // 計測開始時間
					for (auto bias_factor: bias_set) {
						std::vector<std::vector<double>> opt_nmse(alpha_step * sigma_step, std::vector<double>(teacher_signals[TRAIN].size(), 1e+10));
						std::vector < std::vector<double>> opt_lm2(alpha_step * sigma_step, std::vector<double>(teacher_signals[TRAIN].size()));
						std::vector < std::vector <std::vector<double>>> opt_w(alpha_step * sigma_step, std::vector <std::vector<double>>(teacher_signals[TRAIN].size()));
#pragma omp parallel for num_threads(48)
						// 複数のリザーバーの時間発展をまとめて処理
						for (int k = 0; k < alpha_step * sigma_step; k++) {
							const double input_signal_factor = alpha_set[(k / sigma_step)];
							const double weight_factor = sigma_set[(k % sigma_step)];

							reservoir_layer reservoir_layer1(unit_size, unit_size / 10, input_signal_factor, weight_factor, bias_factor, p, nonlinear, loop, wash_out);
							reservoir_layer1.generate_reservoir();

							reservoir_layer1.reservoir_update(input_signal[TRAIN], output_node[k][TRAIN], step);
							reservoir_layer1.reservoir_update(input_signal[VAL], output_node[k][VAL], step);
							reservoir_layer1.reservoir_update(input_signal[TEST], output_node[k][TEST], step);
							is_echo_state_property[k] = reservoir_layer1.is_echo_state_property(input_signal[VAL]);
							reservoir_layer_v[k] = reservoir_layer1;
						}
						int lm;

						int opt_k = 0;

						output_learning output_learning[341];
#pragma omp parallel for num_threads(48)
						// 重みの学習を行う
						for (int k = 0; k < alpha_step * sigma_step; k++) {
							if (!is_echo_state_property[k]) continue;
							output_learning[k].generate_simultaneous_linear_equationsA(output_node[k][TRAIN], wash_out, step, unit_size);

						}
						int i;
#pragma omp parallel for  private(lm, i) num_threads(48)
						for (int k = 0; k < alpha_step * sigma_step; k++) {
							if (!is_echo_state_property[k]) continue;
							std::cerr << k << std::endl;
							std::vector<double> output_node_T, output_node_N;
							for (int i = 0; i <= unit_size; i++) {
								for (int t = wash_out + 1; t < step; t++) {
									output_node_T.push_back(output_node[k][TRAIN][t + 1][i]);
								}
							}
							for (int t = 0; t < step; t++) {
								for (int i = 0; i <= unit_size; i++) {
									output_node_N.push_back(output_node[k][VAL][t + 1][i]);
								}
							}
							for (i = 0; i < teacher_signals[TRAIN].size(); i++) {
								output_learning[k].generate_simultaneous_linear_equationsb_fast(output_node_T, teacher_signals[TRAIN][i], wash_out, step, unit_size);
								double opt_lm = 0;
								double opt_lm_nmse = 1e+9;
								for (lm = 0; lm < lambda_step; lm++) {
									for (int j = 0; j <= unit_size; j++) {
										output_learning[k].A[j][j] += pow(10, -15 + lm * 2);
										if (lm != 0) output_learning[k].A[j][j] -= pow(10, -15 + (lm - 1) * 2);
									}
									output_learning[k].IncompleteCholeskyDecomp2(unit_size + 1);
									double eps = 1e-12;
									int itr = 10;
									output_learning[k].ICCGSolver(unit_size + 1, itr, eps);
									w[k][i][lm] = output_learning[k].w;
									nmse[k][i][lm] = calc_nmse(teacher_signals[VAL][i], output_learning[k].w, output_node[k][VAL], unit_size, wash_out, step, false);
								}
							}
						}

						// 検証データでもっとも性能の良いリザーバーを選択
						for (int k = 0; k < alpha_step * sigma_step; k++) {
							if (!is_echo_state_property[k]) continue;
							std::cerr << "*" << k << std::endl;
							for (int i = 0; i < teacher_signals[TRAIN].size(); i++) {
								for (int lm = 0; lm < lambda_step; lm++) {
									if (nmse[k][i][lm] < opt_nmse[k][i]) {
										opt_nmse[k][i] = nmse[k][i][lm];
										opt_lm2[k][i] = lm;
										opt_w[k][i] = w[k][i][lm];
									}
								}
							}
						}
						std::vector<double> L(alpha_step * sigma_step);
						std::vector<double> train_L(alpha_step * sigma_step);
						std::vector<double> NL(alpha_step * sigma_step);
						std::vector<double> NL_0(alpha_step * sigma_step);
						std::vector<std::vector<double>> sub_NL(alpha_step * sigma_step, std::vector<double>(30));
						std::vector<double> train_NL(alpha_step * sigma_step);
						std::vector<std::vector<double>> narma_task(alpha_step * sigma_step);
						std::vector<std::vector<double>> approx_task(alpha_step * sigma_step);
						#pragma omp parallel for  private(i) num_threads(48)
						for (int k = 0; k < alpha_step * sigma_step; k++) {
							if (!is_echo_state_property[k]) continue;
							for (i = 0; i < teacher_signals[TEST].size(); i++) {
								const double test_nmse = calc_nmse(teacher_signals[TEST][i], opt_w[k][i], output_node[k][TEST], unit_size, wash_out, step, false);
								const double train_nmse = calc_nmse(teacher_signals[TRAIN][i], opt_w[k][i], output_node[k][TRAIN], unit_size, wash_out, step, false);
								if (i < task_size[0]) narma_task[k].push_back(test_nmse);
								else if(i < task_size[1]) approx_task[k].push_back(test_nmse);
								else if (i < task_size[2]) {
									const double tmp_L = 1.0 - test_nmse;
									if (tmp_L >= TRUNC_EPSILON) L[k] += tmp_L;
									const double tmp_train_L = 1.0 - train_nmse;
									if (tmp_train_L >= TRUNC_EPSILON) train_L[k] += tmp_train_L;
								}
								else {
									int d_sum = 0;
									for (auto& e : d_vec[TEST][i - task_size[2]]) d_sum += e;
									const double tmp_NL = d_sum * (1.0 - test_nmse);
									if (tmp_NL >= TRUNC_EPSILON) {
										NL[k] += tmp_NL;
										NL_0[k] += (1.0 - test_nmse);

										sub_NL[k][d_sum] += tmp_NL;
									}
									const double tmp_train_NL = d_sum * (1.0 - train_nmse);
									if (tmp_train_NL >= TRUNC_EPSILON) train_NL[k] += tmp_train_NL;
								}
							}

						}
						
						for (int k = 0; k < alpha_step * sigma_step; k++) {
							if (!is_echo_state_property[k]) continue;
							const double input_signal_factor = alpha_set[(k / sigma_step)];
							const double weight_factor = sigma_set[(k % sigma_step)];
							outputfile << function_name << "," << loop << "," << unit_size << "," << p << "," << bias_factor << "," << input_signal_factor << "," << weight_factor << "," <<
								train_L[k] << "," << L[k] << "," << train_NL[k] << "," << NL[k] << "," << NL_0[k];
							for (int i = 2; i < 30; i++) {
								outputfile << "," << sub_NL[k][i];
							}
							for (int i = 0; i < narma_task[k].size(); i++) {
								outputfile << "," << narma_task[k][i];
							}
							for (int i = 0; i < approx_task[k].size(); i++) {
								outputfile << "," << approx_task[k][i];
							}
							outputfile << std::endl;
						}
					}
				}
			}
		}
		outputfile.close();
	}
}