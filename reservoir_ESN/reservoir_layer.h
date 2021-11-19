//#pragma once
#include <iostream>
#include <algorithm>
#include <cassert>
#include <random>
#include <vector>
#include <numeric>
#include <cassert>
#include <fstream>
#include <string>
#define LINEAR (0)
#define NON_LINEAR (1)
#include <cmath>
//#include <limits>

class reservoir_layer {
public:
	unsigned int unit_size;								//�@�m�[�h��
	//unsigned int connection_degree;						//	1���j�b�g������̐ڑ���	(�m�[�h����1�����x�Ő��x���O�a����j
	double input_signal_factor;							//	���͂̋���
	//double weight_factor;								//	���j�b�g�Ԑڑ��̋����@
	//double bias_factor;									//	�o�C�A�X�̏d�݋���
	//std::vector<std::vector<double>> weight_reservoir;	//  ���U�[�o�[�w�̌����d�݁@
	//std::vector<std::vector<int>> adjacency_list;		//  �O���t�ɂ�����אڃ��X�g(�אڃ��X�g:https://qiita.com/drken/items/4a7869c5e304883f539b)
	std::vector<double> input_signal_strength;			//  ���͑w�̏d�݌����̋����x�N�g��  ����̓}�X�N�M����input_signal_factor�����������̂Ƃ���
	unsigned int seed;									//	���U�[�o�[�̍\�������肷��V�[�h�l�i�\���̃V�[�h�Əd�݂̃V�[�h�Ȃǂ̕��������Ă����������j
	double p;											//	�m�[�h�Ŏg�p����銈�����֐��̔���`�̊���
	//double (*nonlinear)(double);	//	����`�֐��̊֐��|�C���^
	double(*nonlinear)(double, double, double, double);
	std::vector<int> node_type;							//	n�Ԗڂ̃m�[�h�̐��`/����`�̎��
	std::mt19937 mt;
	int wash_out;
	double input_gain;
	double feed_gain;
	//std::vector<double> Mask;  ����͕K�v�Ȃ�

	std::vector<std::vector<double>> J;
	//double pa;

	std::vector<double> a;
	std::vector<double> b;
	double j;
	double j2;



	reservoir_layer();
	//reservoir_layer(const int unit_size, const int connection_degree, const double iss_factor, const double weight_factor, const double bias_factor, const double p,
		//double (*nonlinear)(double), const unsigned int seed, const int wash_out);
	reservoir_layer (const int unit_size, const double iss_factor, const double input_gain, const double feed_gain, const double p, double (*nonlinear)(double, double, double, double),//�ύX����
		const unsigned int seed, const int wash_out, const int t_size);

	void generate_reservoir();
	void reservoir_update(const std::vector<double>& input_signal, std::vector<std::vector<double>>& output_node, const int t_size, int seed = 0);
	void reservoir_update_show(const std::vector<double> input_signal, std::vector<std::vector<double>> output_node, const int t_size, const int wash_out, const std::string name);
	bool is_echo_state_property(const std::vector<double>& input_signal);
	double activation_function(const double x1, const double x2, const int type, const double J);
	double activation_function2(const double x1, const double x2, const double x3, const int type, const double J);
};





