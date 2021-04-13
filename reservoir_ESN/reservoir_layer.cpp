#include "reservoir_layer.h"
reservoir_layer::reservoir_layer() {}
reservoir_layer::reservoir_layer(const int unit_size, const int connection_degree, const double iss_factor, const double weight_factor, const double p,
	double (*nonlinear)(double), const unsigned int seed = 0, const int wash_out = 500) {
	this->unit_size = unit_size;
	this->connection_degree = connection_degree;
	this->input_signal_strength = iss_factor;
	this->weight_factor = weight_factor;
	this->p = p;
	this->seed = seed;
	this->nonlinear = nonlinear;
	this->wash_out = wash_out;
	node_type.resize(unit_size + 1);
	adjacency_list.resize(unit_size + 1, std::vector<int>(connection_degree + 1));
	weight_reservoir.resize(unit_size + 1, std::vector<double>(connection_degree + 1));
	input_signal_strength.resize(unit_size + 1);
	mt.seed(seed);
}

void reservoir_layer::generate_reservoir() {
	std::uniform_real_distribution<> rand_minus1toplus1(-1, 1);
	std::uniform_int_distribution<> rand_0or1(0, 1);

	std::vector<int> permutation(unit_size);
	std::iota(permutation.begin(), permutation.end(), 1);
	//���U�[�o�[�w�̌����������_���ɐ���
	for (int n = 1; n <= unit_size; n++) {
		std::shuffle(permutation.begin(), permutation.end(), mt);
		for (int k = 1; k <= connection_degree; k++) {
			adjacency_list[n][k] = permutation[k];
		}
	}

	//�e�m�[�h�����`������`�������� 0->���` 1->����`
	for (int n = 1; n <= unit_size; n++) {
		if (permutation[n] <= unit_size * p)
			node_type[n] = NON_LINEAR;
		else
			node_type[n] = LINEAR;
	}

	for (int n = 1; n <= unit_size; n++) {
		//���U�[�o�[�w�̌����d�݂�����
		weight_reservoir[n][0] = input_signal_factor * weight_factor * rand_minus1toplus1(mt);
		for (int k = 1; k <= connection_degree; k++)
			weight_reservoir[n][k] = weight_factor * (1.0 / sqrt(connection_degree)) * (rand_0or1(mt) * 2 - 1);

		// ���͑w�̌����d�݂�����
		input_signal_strength[n] = weight_factor * input_signal_factor * (rand_0or1(mt) * 2 - 1);
	}
}
/** ���U�[�o�[�w�����Ԕ��W������
	 * input_signal ���͐M��
	 * output_node[t][n] ����t�ɂ�����n�Ԗڂ̃m�[�h�̏o��
	 * t_size �X�e�b�v��
	 **/
void reservoir_layer::reservoir_update(const std::vector<double>& input_signal, std::vector<std::vector<double>>& output_node, const int t_size) {
	std::uniform_real_distribution<> rand_minus1toplus1(-1, 1);

	output_node[0][0] = 1.0;
	for (int n = 1; n <= unit_size; n++) output_node[0][n] = rand_minus1toplus1(mt);

	std::vector<double> input_sum_node(unit_size + 1, 0);
	for (int t = 0; t <= t_size; t++) {
		for (int n = 1; n <= unit_size; n++) {
			input_sum_node[n] = input_signal_strength[n] * input_signal[t];
			for (int k = 1; k <= connection_degree; k++) input_sum_node[n] += weight_reservoir[n][k] * output_node[t][adjacency_list[n][k]];
			input_sum_node[n] += weight_reservoir[n][0] * output_node[t][0];
		}
		output_node[t + 1][0] = 1.0;
		for (int n = 1; n <= unit_size; n++) output_node[t + 1][n] = activation_function(input_sum_node[n], node_type[n]);
	}
}
// ESP(Echo State Property)�̗L�����`�F�b�N����
bool reservoir_layer::ESP_check(const std::vector<double>& input_signal, std::vector<std::vector<double>>& output_node) {
	auto dummy_output_node = std::vector<std::vector<double>>(wash_out + 1, std::vector<double>(unit_size + 1, 0));
	dummy_output_node[0][0] = 1.0;
	for (int n = 1; n <= unit_size; n++) output_node[0][n] = 1.0;

	// ESP�̃`�F�b�N���Cdummy���ǂ̂悤�ȏ�����ԂŎn�߂邩������@�Ƃ肠�����͑S�m�[�h1�ɂ��Ă���
	// for(int n = 1; n <= unit_size; n++) dummy_output_node[0][n] = rand_minus1toplus1(mt);

	std::vector<double> input_sum_node(unit_size + 1, 0);
	for (int t = 0; t <= wash_out; t++) {
		for (int n = 1; n <= unit_size; n++) {
			input_sum_node[n] = input_signal_strength[n] * input_signal[t];
			for (int k = 1; k <= connection_degree; k++) input_sum_node[n] += weight_reservoir[n][k] * dummy_output_node[t][adjacency_list[n][k]];
			input_sum_node[n] += weight_reservoir[n][0] * dummy_output_node[t][0];
		}
		dummy_output_node[t + 1][0] = 1.0;
		for (int n = 1; n <= unit_size; n++) dummy_output_node[t + 1][n] = activation_function(input_sum_node[n], node_type[n]);
	}

	double err_sum = 0.0;
	for (int t = wash_out - 9; t <= wash_out; t++) {
		for (int n = 1; n <= unit_size; n++) {
			double err = std::abs(output_node[t][n] - dummy_output_node[t][n]);
			err_sum += err * err;
		}
	}
	double err_ave = err_sum / (unit_size * 10);
	if (err_ave <= 0.1) {
		return true;
	}
	else {
		return false;
	}
}

double reservoir_layer::activation_function(const double x, const int type) {
	if (type == LINEAR) {
		return std::max(-100.0, std::min(100.0, x));
	}
	else if (type == NON_LINEAR) {
		return nonlinear(x);
	}
	assert(type != LINEAR && type != NON_LINEAR);
	return -1.0;
}