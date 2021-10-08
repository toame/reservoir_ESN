#include "task.h"
#include <cblas.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
void generate_input_signal_random(std::vector<double>& input_signal, const int u_min, const int u_delta, const int step, const int seed) {
	std::mt19937 mt(seed);
	std::uniform_real_distribution<> rand_0to1(0, 1);
	std::uniform_int_distribution<> rand_0or1(0, 1);
	for (int t = 0; t < step; t++) input_signal.push_back(u_min + u_delta * rand_0to1(mt));
}

void task_for_function_approximation(const std::vector<double>& input_signal, std::vector<double>& output_signal, const double nu, const int tau,
	const int step, const int seed) {
	std::mt19937 mt(seed);
	std::uniform_real_distribution<> rand_0to1(0, 1);
	std::uniform_int_distribution<> rand_0or1(0, 1);

	for (int t = 0; t < step; t++) {
		if (t - tau >= 0)
			output_signal.push_back(sin(nu * PI * input_signal[t - tau]));
		else
			output_signal.push_back(0);
	}
}
//  0.3, 0.05, 1.5, 0.1
void generate_narma_task(std::vector<double>& input_signal, std::vector<double>& teacher_signal, int tau, int step) {
	const double alpha = 0.3;
	const double beta = 0.05;
	const double gamma = 1.5;
	double delta = 0.1;
	tau--;
	if (tau >= 10) delta = 0.01;
	for (int t = 0; t < step; t++) {
		input_signal[t] = (input_signal[t] + 1) / 4;
	}
	teacher_signal.resize(step);
	for (int t = 0; t < step; t++) {
		double sum = 0.0;
		if (t - tau >= 0) {
			for (int i = tau + 1; i >= 1; i--) {  // for(i=tau; i>=1; i--){
				sum = sum + teacher_signal[t - i];
			}

			teacher_signal[t] =
				alpha * teacher_signal[t - 1] + beta * teacher_signal[t - 1] * sum + gamma * input_signal[t - tau] * input_signal[t] + delta;
			if (tau > 9) teacher_signal[t] = tanh(teacher_signal[t]);  // NARMA(tau>=10)
		}
		else
			teacher_signal[t] = 0;

		//... cutt-off bound ...
		if (teacher_signal[t] > 1.0) {
			teacher_signal[t] = 1.0;
		}
		else if (teacher_signal[t] < -1.0) {
			teacher_signal[t] = -1.0;
		}
	}
}

//  0.3, 0.05, 1.5, 0.1
void generate_narma_task2(std::vector<double> input_signal, std::vector<double>& teacher_signal, int tau, int step) {
	const double alpha = 0.3;
	const double beta = 0.05;
	const double gamma = 1.5;
	double delta = 0.1;
	tau--;
	if (tau >= 10) delta = 0.01;
	for (int t = 0; t < step; t++) {
		input_signal[t] = (input_signal[t] + 1) / 4;
	}
	teacher_signal.resize(step);
	for (int t = 0; t < step; t++) {
		double sum = 0.0;
		if (t - tau >= 0) {
			for (int i = tau + 1; i >= 1; i--) {  // for(i=tau; i>=1; i--){
				sum = sum + teacher_signal[t - i];
			}

			teacher_signal[t] =
				alpha * teacher_signal[t - 1] + beta * teacher_signal[t - 1] * sum + gamma * input_signal[t - tau] * input_signal[t] + delta;
			if (tau > 9) teacher_signal[t] = tanh(teacher_signal[t]);  // NARMA(tau>=10)
		}
		else
			teacher_signal[t] = 0;

		//... cutt-off bound ...
		if (teacher_signal[t] > 1.0) {
			teacher_signal[t] = 1.0;
		}
		else if (teacher_signal[t] < -1.0) {
			teacher_signal[t] = -1.0;
		}
	}
}

void generate_input_signal_henon_map(std::vector<double>& input_signal, const int fstep, const int step, const int wash_out) {
	double a = 0.1, b = 0.2, c = 0;
	const double alpha = 1.4;
	const double beta = 0.3;
	input_signal.resize(step + fstep + 10);
	for (int t = 0; t < wash_out; t++) {
		c = 1 - alpha * b * b + beta * a;
		std::swap(a, b);
		std::swap(b, c);
	}
	input_signal[0] = a;
	input_signal[1] = b;
	for (int t = 2; t <= step + fstep; t++) {
		input_signal[t] = 1 - alpha * input_signal[t - 1] * input_signal[t - 1] + beta * input_signal[t - 2];
	}
}

void generate_henom_map_task(std::vector<double>& input_signal, std::vector<double>& teacher_signal, const int fstep, const int step, const int wash_out) {
	generate_input_signal_henon_map(input_signal, fstep, step, wash_out);
	teacher_signal.resize(step);
	for (int t = 0; t < step; t++) {
		teacher_signal[t] = input_signal[t + fstep];
	}
}

void generate_input_signal_wave(std::vector<double>& input_signal, const double nu, const int step, const int wash_out) {
	for (int t = 0; t < step; t++) {
		input_signal[t] = sin(nu * t + wash_out) + sin(nu * t / 10.0 + wash_out);
	}
}

void generate_input_signal_wave(std::vector<double>& input_signal, std::vector<double>& teacher_signal, const int fstep, const int step, const int wash_out) {
	generate_input_signal_henon_map(input_signal, fstep, step, wash_out);
	teacher_signal.resize(step);
	for (int t = 0; t < step; t++) {
		teacher_signal[t] = input_signal[t + fstep];
	}
}

void generate_input_signal_laser(std::vector<double>& input_signal, const int fstep, const int step, const int wash_out) {
	std::ifstream ifs("santafe.dat");

	std::string line;
	int cnt = wash_out;
	while (std::getline(ifs, line)) {
		cnt--;
		if (cnt <= 0) {
			double num = std::stoi(line) / 255.0;
			//std::cerr << num << std::endl;
			input_signal.push_back(num);
		}
	}
}

void generate_laser_task(std::vector<double>& input_signal, std::vector<double>& teacher_signal, const int fstep, const int step, const int wash_out) {
	generate_input_signal_laser(input_signal, fstep, step, wash_out);
	teacher_signal.resize(step);
	for (int t = 0; t < step; t++) {
		teacher_signal[t] = input_signal[t + fstep];
	}
}

void task_for_calc_of_L(const std::vector<double>& input_signal, std::vector<double>& teacher_signal, const int tau, const int step) {
	teacher_signal.resize(step);
	for (int t = 0; t < step; t++) {
		if (t - tau >= 0)
			teacher_signal[t] = input_signal[t - tau];
		else
			teacher_signal[t] = 0.0;
	}
}
void generate_d_sequence(std::vector<std::vector<int>>& d_vec, std::vector<int>& d, int d_sum_remain, int depth = 0) {
	if (d_sum_remain <= 0) {
		d_vec.push_back(d);
		// for(auto& e: d) std::cerr << e << " ";
		// std::cerr << std::endl;
		return;
	}
	// std::cerr << d_sum_remain << " " << depth << std::endl;
	d[depth]++;
	generate_d_sequence(d_vec, d, d_sum_remain - 1, depth);
	d[depth]--;
	if (depth + 1 < d.size()) generate_d_sequence(d_vec, d, d_sum_remain, depth + 1);
}

void generate_d_sequence_set(std::vector<std::vector<std::vector<int>>>& d_vec) {
	std::vector<int> d;
	const int mode = 0;
	d.resize(8); generate_d_sequence(d_vec[mode], d, 2);
	d.resize(8); generate_d_sequence(d_vec[mode], d, 3);
	d.resize(7); generate_d_sequence(d_vec[mode], d, 4);
	d.resize(6); generate_d_sequence(d_vec[mode], d, 5);
	d.resize(5); generate_d_sequence(d_vec[mode], d, 6);
	d.resize(4); generate_d_sequence(d_vec[mode], d, 7);
	d.resize(3); generate_d_sequence(d_vec[mode], d, 8);
	for (int u = 9; u < 30; u++) {
		d.resize(2);
		generate_d_sequence(d_vec[mode], d, u);
	}
}

void task_for_calc_of_NL(const std::vector<double>& input_signal, std::vector<double>& output_signal, std::vector<int>& d, const int step) {
	for (int t = 0; t <= step; t++) {
		double x = 1.0;
		for (int i = 0; i < d.size(); i++) {
			if (t - (i + 1) >= 0) x *= std::legendre(d[i], input_signal[t - (i + 1)]);
		}
		output_signal[t] = x;
	}
}

inline double squared(const double x) {
	return x * x;
}

double t_tt_calc(std::vector<double> yt, const int wash_out, const int step) {
	double t_ave0 = 0.0, tt_ave0 = 0.0;
	for (int t = wash_out + 1; t < step; t++) {
		t_ave0 += yt[t];
		tt_ave0 += yt[t] * yt[t];
	}
	t_ave0 /= (step - wash_out);
	tt_ave0 /= (step - wash_out);
	return tt_ave0 - t_ave0 * t_ave0;
}

double calc_mean_squared_average(const std::vector<double>& teacher_signal, const std::vector<double>& weight,
	const std::vector<std::vector<double>>& output_node, const int unit_size, const int wash_out, const int step, bool show, std::string name) {
	double sum_squared_average = 0.0;
	//std::ofstream outputfile("output_predict/" + name + ".txt", std::ios::app);
	//if(show)
	//	outputfile << "t,predict_test,teacher" << std::endl;
	for (int t = wash_out + 1; t < step; t++) {
		const double reservoir_predict_signal = cblas_ddot(unit_size + 1, weight.data(), 1, output_node[t + 1].data(), 1);
		//double reservoir_predict_signal = 0.0;
		//for (int n = 0; n <= unit_size; n++) {
		//	reservoir_predict_signal += weight[n] * output_node[t + 1][n];
		//}
		sum_squared_average += squared(teacher_signal[t] - reservoir_predict_signal);
		//if (show) {
		//	outputfile << t << "," << reservoir_predict_signal << "," << teacher_signal[t] << "," << sum_squared_average << std::endl;
		//}
	}
	return sum_squared_average / (step - wash_out);
}

double calc_nmse(const std::vector<double>& teacher_signal, const std::vector<double>& weight,
	const std::vector<std::vector<double>>& output_node, const int unit_size, const int wash_out, const int step, bool show, std::string name) {
	return (calc_mean_squared_average(teacher_signal, weight, output_node, unit_size, wash_out, step, show, name) / t_tt_calc(teacher_signal, wash_out, step));
}