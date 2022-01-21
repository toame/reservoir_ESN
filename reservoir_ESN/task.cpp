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

	/*//�m�C�Y����̏ꍇ
	std::random_device seed_gen;
	std::default_random_engine engine(seed_gen());
	std::normal_distribution<> dist(0.0, 1.0);
	std::mt19937 mt2(1);
	const double �� = 0.001;*/

	for (int t = 0; t < step; t++) {
		if (t - tau >= 0)
			output_signal.push_back(sin(nu * PI * input_signal[t - tau]));
		else
			output_signal.push_back(0);
	}

	/*for (int t = 0; t < step; t++) {
		output_signal[t] += dist(mt2) * ��;
	}*/
}

void task_for_function_approximation2(const std::vector<double>& input_signal, std::vector<double>& output_signal, const double nu, const int tau,
	const int step, const int seed) {
	std::mt19937 mt(seed);
	std::uniform_real_distribution<> rand_0to1(0, 1);
	std::uniform_int_distribution<> rand_0or1(0, 1);

	//for (int t = 0; t < step; t++) {
	//	if (t - tau >= 0) {
	//		//output_signal.push_back(sin(PI * nu * input_signal[t - 1]));
	//		output_signal.push_back(sin(PI * nu * input_signal[t]) * output_signal[t - tau] + 0.5);
	//	}
	//	else
	//		output_signal.push_back(0);
	//}
	for (int t = 0; t < step; t++) {
		double sum = 0;
		for (int r = std::max(0, t - tau); r <= t; r++) sum += input_signal[r];
		sum /= sqrt(tau + 1);
		output_signal.push_back(sin(nu * PI * sum));
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
		if (t - tau - 1>= 0) {
			for (int i = tau + 1; i >= 1; i--) {  // for(i=tau; i>=1; i--){
				sum = sum + teacher_signal[t - i];
			}

			teacher_signal[t] =
				alpha * teacher_signal[t - 1] + beta * teacher_signal[t - 1] * sum + gamma * input_signal[t - tau - 1] * input_signal[t] + delta;
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
		if (t - tau - 1 >= 0) {
			for (int i = tau + 1; i >= 1; i--) {  // for(i=tau; i>=1; i--){
				sum = sum + teacher_signal[t - i];
			}

			teacher_signal[t] =
				alpha * teacher_signal[t - 1] + beta * teacher_signal[t - 1] * sum + gamma * input_signal[t - tau - 1] * input_signal[t] + delta;
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

void generate_input_signal_henon_map2(std::vector<double>& input_signal, const int fstep, const int step, const int wash_out) {
	std::random_device seed_gen;
	std::default_random_engine engine(seed_gen());
	std::normal_distribution<> dist(0.0, 1.0);
	std::mt19937 mt(2);
	double a = 0.1, b = 0.2, c = 0;
	const double alpha = 1.4;
	const double beta = 0.3;
	const double �� = 0.001;
	//std::vector<double> ��;
	/*for (int t = 2; t <= step + fstep; t++) {
		��[t] = �� * dist(engine);
	}*/
	input_signal.resize(step + fstep + 10);
	for (int t = 0; t < wash_out; t++) {
		c = 1 - alpha * b * b + beta * a;
		std::swap(a, b);
		std::swap(b, c);
	}
	input_signal[0] = a;
	input_signal[1] = b;
	for (int t = 2; t <= step + fstep; t++) {
		input_signal[t] = 1.0 - alpha * input_signal[t - 1] * input_signal[t - 1] + beta * input_signal[t - 2];// +dist(mt) * ��;
	}

	for (int t = 0; t <= step + fstep; t++) {
		input_signal[t] += dist(mt)*��;
	}
	/*for (int t = 0; t <= step + fstep; t++) {
		input_signal[t] *= ��;
	}*/
}

void generate_input_signal_henon_map3(std::vector<double>& input_signal, const int fstep, const int step, const int wash_out) {
	std::random_device seed_gen;
	std::default_random_engine engine(seed_gen());
	std::normal_distribution<> dist(0.0, 1.0);
	std::mt19937 mt(2);
	double a = 0.1, b = 0.2, c = 0;
	const double alpha = 1.4;
	const double beta = 0.3;
	const double �� = 0.001;
	//std::vector<double> ��;
	/*for (int t = 2; t <= step + fstep; t++) {
		��[t] = �� * dist(engine);
	}*/
	input_signal.resize(step + fstep + 10);
	for (int t = 0; t < wash_out; t++) {
		c = 1 - alpha * b * b + beta * a;
		std::swap(a, b);
		std::swap(b, c);
	}
	input_signal[0] = a;
	input_signal[1] = b;
	for (int t = 2; t <= step + fstep; t++) {
		input_signal[t] = 1.0 - alpha * input_signal[t - 1] * input_signal[t - 1] + beta * input_signal[t - 2];// +dist(mt) * ��;
	}


	for (int t = 0; t <= step + fstep; t++) {
		if (input_signal[t] < 0) {
			input_signal[t] *= -1.0;
		}
	}

	for (int t = 0; t <= step + fstep; t++) {
		input_signal[t] *= 100.0;
	}
	/*for (int t = 0; t <= step + fstep; t++) {
		input_signal[t] *= ��;
	}*/
}


void generate_henom_map_task(std::vector<double>& input_signal, std::vector<double>& teacher_signal, const int fstep, const int step, const int wash_out) {
	generate_input_signal_henon_map(input_signal, fstep, step, wash_out);
	teacher_signal.resize(step);
	for (int t = 0; t < step; t++) {
		teacher_signal[t] = input_signal[t + fstep];
	}
}

void generate_henom_map_task3(std::vector<double>& input_signal, std::vector<double>& teacher_signal, const int fstep, const int step, const int wash_out) {
	generate_input_signal_henon_map3(input_signal, fstep, step, wash_out);
	teacher_signal.resize(step);
	for (int t = 0; t < step; t++) {
		teacher_signal[t] = input_signal[t + fstep];
	}
}

void generate_henom_map_task2(std::vector<double>& input_signal, std::vector<double>& teacher_signal, const int fstep, const int step, const int wash_out) {
	generate_input_signal_henon_map2(input_signal, fstep, step, wash_out);
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

void generate_legendre_task(const std::vector<double>& input_signal, std::vector<double>& teacher_signal, int nu, int tau, const int step) {
	teacher_signal.resize(step);
	for (int t = 0; t < step; t++) {
		double x = 1.0;
		if (t - tau >= 0) x *= std::legendre(nu, input_signal[t - tau]);
		teacher_signal[t] = x;
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

inline double squared(const double x) {
	return x * x;
}

double t_tt_calc(std::vector<double> yt, const int wash_out, const int step) {//https://www.osapublishing.org/osac/fulltext.cfm?uri=osac-4-3-1086&id=449229
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
	std::ofstream outputfile("output_predict_STDE/" + name + ".txt", std::ios::app);//predict �\��
	if (show)
		outputfile << "t,predict_test,teacher" << std::endl;
	for (int t = wash_out + 1; t < step; t++) {
		//const double reservoir_predict_signal = cblas_ddot(unit_size + 1, weight.data(), 1, output_node[t + 1].data(), 1);
		double reservoir_predict_signal = 0.0;
		for (int n = 0; n <= unit_size; n++) {
			reservoir_predict_signal += weight[n] * output_node[t + 1][n];//��O
		}
		sum_squared_average += squared(teacher_signal[t] - reservoir_predict_signal);
		//sum_squared_average = sqrt(sum_squared_average);
		if (show) {
			outputfile << t << "," << reservoir_predict_signal << "," << teacher_signal[t] << "," << sum_squared_average << std::endl;
		}
	}
	return sum_squared_average / (step - wash_out);
}

double calc_nmse(const std::vector<double>& teacher_signal, const std::vector<double>& weight,
	const std::vector<std::vector<double>>& output_node, const int unit_size, const int wash_out, const int step, bool show, std::string name) {
	//return (sqrt(calc_mean_squared_average(teacher_signal, weight, output_node, unit_size, wash_out, step, show, name) / t_tt_calc(teacher_signal, wash_out, step)));
	return (calc_mean_squared_average(teacher_signal, weight, output_node, unit_size, wash_out, step, show, name) / t_tt_calc(teacher_signal, wash_out, step));
}

double calc_nrmse(const std::vector<double>& teacher_signal, const std::vector<double>& weight,
	const std::vector<std::vector<double>>& output_node, const int unit_size, const int wash_out, const int step, bool show, std::string name) {
	double sum_squared_average = 0.0;
	double sum_squared_predict_signal = 0.0;
	double y_max = -1e+9;
	double y_min = 1e+9;
	std::ofstream outputfile("output_predict/" + name + ".txt", std::ios::app);//predict �\��
	if (show)
		outputfile << "t,predict_test,teacher" << std::endl;
	for (int t = wash_out + 1; t < step; t++) {
	//for (int t = 1; t < step; t++) {
		//const double reservoir_predict_signal = cblas_ddot(unit_size + 1, weight.data(), 1, output_node[t + 1].data(), 1);
		double reservoir_predict_signal = 0.0;
		for (int n = 0; n <= unit_size; n++) {
			reservoir_predict_signal += weight[n] * output_node[t + 1][n];//��O
		}
		y_max = std::max(y_max, reservoir_predict_signal);
		y_min = std::min(y_min, reservoir_predict_signal);
		sum_squared_average += squared(teacher_signal[t] - reservoir_predict_signal);
		sum_squared_predict_signal += squared(reservoir_predict_signal);//�Ƃ肠����
		//sum_squared_average = sqrt(sum_squared_average);
		if (show) {
			outputfile << t << "," << reservoir_predict_signal << "," << teacher_signal[t] << "," << sum_squared_average << std::endl;
		}
	}

	const double mse = sum_squared_average / (step - wash_out);//�{���͂�����
	//const double mse = sum_squared_average / sum_squared_predict_signal;//�Ƃ肠����
	//const double rmse = sqrt(mse);

	return sqrt(mse / squared((y_max - y_min)));//�{���͂�����
	//return rmse;//�ύX�\��@https://www.sciencedirect.com/science/article/pii/S1463500313001418�@1���͂��

}
/*double calc_nrmse(const std::vector<double>& teacher_signal, const std::vector<double>& weight,
	const std::vector<std::vector<double>>& output_node, const int unit_size, const int wash_out, const int step, bool show, std::string name) {
	//return (sqrt(calc_mean_squared_average(teacher_signal, weight, output_node, unit_size, wash_out, step, show, name) / t_tt_calc(teacher_signal, wash_out, step)));
	return sqrt((calc_mean_squared_average(teacher_signal, weight, output_node, unit_size, wash_out, step, show, name) / t_tt_calc(teacher_signal, wash_out, step)));
}*/