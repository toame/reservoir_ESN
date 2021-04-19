#include <iostream>
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
int main(void) {

	const int unit_size = 200;
	const int step = 4000;
	const int wash_out = 500;
	std::vector<std::vector<double>> input_signal(PHASE_NUM), teacher_signal(PHASE_NUM);
	const std::string task_name = "APPROX";
	double alpha_min, d_alpha;
	for (int phase = 0; phase < PHASE_NUM; phase++) {
		generate_input_signal_random(input_signal[phase], -1.0, 2.0, step, phase + 100);
		if (task_name == "NARMA") {
			d_alpha = 0.01;
			const int tau = 9;
			generate_narma_task(input_signal[phase], teacher_signal[phase], tau, step);
		}
		else if(task_name == "APPROX") {
			d_alpha = 1.0;
			alpha_min = 0.2;
			const double nu = 1.5;
			const int tau = 5;
			task_for_function_approximation(input_signal[phase], teacher_signal[phase], nu, tau, step, phase);

		}
		
	}
	//for (int i = 0; i < input_signal[0].size(); i++) {
	//	std::cout << input_signal[0][i] << " " << teacher_signal[0][i] << std::endl;
	//}
	//return 0;


	std::chrono::system_clock::time_point  start, end; // 型は auto で可
	for (int loop = 0; loop < 1; loop++) {
		std::vector<std::vector<std::vector<std::vector<double>>>> output_node(11 * 11, std::vector<std::vector<std::vector<double>>>(PHASE_NUM, std::vector<std::vector<double>>(step + 2, std::vector<double>(unit_size + 1, 0))));
		std::vector<reservoir_layer> reservoir_layer_v(11 * 11);
		for (int ite_p = 0; ite_p <= 10; ite_p += 1) {
			double opt_nmse = 1e+10;
			double opt_input_signal_factor = 0;
			double opt_weight_factor = 0;
			double opt_lm2 = 0;
			double test_nmse = 1e+10;
			start = std::chrono::system_clock::now(); // 計測開始時間

//#pragma omp parallel for
			for (int k = 0; k < 11 * 11; k++) {

				const double p = ite_p * 0.1;
				const double input_signal_factor = (k / 11) * d_alpha + alpha_min;
				const double weight_factor = (k % 11 + 1) * 0.1;

				reservoir_layer reservoir_layer1(unit_size, unit_size / 10, input_signal_factor, weight_factor, p, sinc, loop, wash_out);
				reservoir_layer1.generate_reservoir();

				reservoir_layer1.reservoir_update(input_signal[TRAIN], output_node[k][TRAIN], step);
				reservoir_layer1.reservoir_update(input_signal[VAL], output_node[k][VAL], step);
				reservoir_layer_v[k] = reservoir_layer1;
			}
			int lm;
			std::vector<std::vector<std::vector<double>>> w(11 * 11, std::vector<std::vector<double>>(10));
			std::vector<std::vector<double>> nmse(11 * 11, std::vector<double>(10));
			int opt_k = 0;
			//#pragma omp parallel for
			for (int k = 0; k < 11 * 11; k++) {
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
						if(lm != 0) output_learning.A[j][j] -= pow(10, -16 + lm - 1);
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

			for (int k = 0; k < 11 * 11; k++) {
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
			reservoir_layer_v[opt_k].reservoir_update(input_signal[TEST], output_node[opt_k][TEST], step);
			test_nmse = calc_nmse(teacher_signal[TEST], opt_w, output_node[opt_k][TEST], unit_size, wash_out, step, false);
			end = std::chrono::system_clock::now();  // 計測終了時間
			double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); //処理に要した時間をミリ秒に変換

			std::cout << ite_p * 0.1 << "," << opt_input_signal_factor << "," << opt_weight_factor << "," << opt_lm2 << "," << opt_nmse << "," << test_nmse << " " << elapsed / 1000.0 << std::endl;
		}
	}
}
