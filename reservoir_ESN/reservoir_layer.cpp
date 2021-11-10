#include "reservoir_layer.h"
reservoir_layer::reservoir_layer() {}
reservoir_layer::reservoir_layer(const int unit_size, const double iss_factor, const double input_gain, const double feed_gain, const double p,
	double (*nonlinear)(double, double, double, double), unsigned int seed = 0, const int wash_out = 500, const int t_size = 3000) {//�ύX����
	this->unit_size = unit_size;
	//this->connection_degree = connection_degree;
	this->input_signal_factor = iss_factor;
	//this->weight_factor = weight_factor;
	//this->bias_factor = bias_factor;
	this->p = p;
	this->seed = seed;
	this->nonlinear = nonlinear;
	this->wash_out = wash_out;
	node_type.resize(unit_size + 1);//???????????
	//adjacency_list.resize(unit_size + 1, std::vector<int>(connection_degree + 1));
	//weight_reservoir.resize(unit_size + 1, std::vector<double>(connection_degree + 1));
	input_signal_strength.resize(unit_size + 1);
	mt.seed(seed);
	this->input_gain = input_gain;
	this->feed_gain = feed_gain;


	//std::vector<std::vector<double>> J; //J�����T�C�Y���Ȃ��ƃ_�������@
	J.resize(t_size + 1, std::vector<double>(unit_size + 1));
	//double pa = 2.0;//�B���@�����Őݒ�
}

// �����g�|���W�[�⌋���d�݂Ȃǂ�ݒ肷��  ���̌�}�X�N�M����邩���o�Q�lor6�l�̃����_���M���p�������5�l�����_���M��
void reservoir_layer::generate_reservoir() {
	 
	std::uniform_real_distribution<> rand_minus1toplus1(-1, 1);//�����_������
	std::uniform_int_distribution<> rand_0or1(-2, 2);//int������0��1

	std::vector<int> permutation(unit_size + 1);      //?�H�H�H�H�H�H�Hpermutation ����@�u��    
	std::iota(permutation.begin(), permutation.end(), 1); //?�H�H�H�H�@https://kaworu.jpn.org/cpp/std::iota

	for (int n = 1; n <= unit_size; n++) {
		std::shuffle(permutation.begin(), permutation.end(), mt); //?https://cpprefjp.github.io/reference/algorithm/shuffle.html
	}
	
 

	//�e�m�[�h�����`������`��������
	for (int n = 1; n <= unit_size; n++) {
		if (permutation[n] <= unit_size * p) {  
			node_type[n] = NON_LINEAR;
		}
		else
			node_type[n] = LINEAR;
	}

	for (int n = 1; n <= unit_size; n++) {
		// ���͑w�̌����d�݂����� �}�X�N�M���Ɠ��͂̋��݂������ňꏏ�ɂ��Ă���
		input_signal_strength[n] = input_signal_factor * (rand_0or1(mt) / 2);
	}
}


/** ���U�[�o�[�w�����Ԕ��W������
	 * input_signal ���͐M��
	 * output_node[t][n] ����t�ɂ�����n�Ԗڂ̃m�[�h�̏o��
	 * t_size �X�e�b�v��
	 **/

void reservoir_layer::reservoir_update(const std::vector<double>& input_signal, std::vector<std::vector<double>>& output_node, const int t_size, int seed) {
	std::mt19937 mt2; // �����Z���k�E�c�C�X�^��32�r�b�g��
	mt2.seed(seed);  
	std::uniform_real_distribution<> rand_minus1toplus1(-1, 1);
	output_node[0][0] = 1.0;//�ύX����v�f
	for (int n = 1; n <= unit_size; n++) output_node[0][n] = rand_minus1toplus1(mt2);
	//std::vector<double> virtual_output_node(unit_size + 1, 0);


	const double e = 2.718;// 281828459045;
	double ��, d;
	d = 95 / (double)unit_size;//���� +1��������  d = �� / N�����݃сi�x�����ԁj��1�Ƃ��Ă��邪�_���ł�80�Ƃ��Ă���ꍇ��������
	/*
	�� = 2�@input_gain��feed_gain�@/50�����@�ق��ɂ��ς��Ă݂�err_sum�@�@�̒l��100�ɂȂ�ɂ͂ǂ�����΂悢���l����@�@�@  err_sum = 3000�t�߁@�@���@���T�̔��\������܂Ƃ߂��������̂́H�H�H
	�с@��21  input_gain feed_gain��0.8���0.1������
	�� = 25��err 0.24
	27�ōX�V
	�� = 95 err_ave  0.1345
	*/
	�� = log(1.0 + d);

	//std::vector<double> input_sum_node(unit_size + 1, 0);    //�v�f��unit_size+1�A�S�Ă̗v�f�̒l0 �ŏ�����

	for (int t = 1; t <= t_size; t++) {
		for (int n = 1; n <= unit_size; n++) {
			J[t][n] = input_signal[t - 1] * input_signal_strength[n];//��O����
		}
	}

	/*for (int t = 1; t <= t_size; t++) {//t = 0��t = 1�ɕύX
		output_node[t][0] = output_node[t - 1][unit_size];
		for (int n = 1; n <= unit_size; n++) {
			//output_node[t][n] = activation_function(output_node[t - 1][n], node_type[n], J[t][n]);//�����̈��������Ƒ����邩��
			output_node[t][n] = activation_function(output_node[t - 1][n], node_type[n]);
			output_node[t][n] *= (1 - pow(e, -��));
			output_node[t][n] += pow(e, -��) * (output_node[t][n - 1]);
		}
	}*/
	for (int t = 1; t <= t_size; t++) {//t = 0��t = 1�ɕύX
		output_node[t][0] = output_node[t - 1][unit_size];
		for (int n = 1; n <= unit_size; n++) {
			output_node[t][n] = activation_function(output_node[t - 1][n], node_type[n], J[t][n]);//�����̈��������Ƒ����邩��
			//output_node[t][n] = activation_function(output_node[t - 1][n], node_type[n]);
			output_node[t][n] *= (1 - pow(e, -��));
			output_node[t][n] += pow(e, -��) * (output_node[t][n - 1]);
		}
	}

}


void reservoir_layer::reservoir_update_show(const std::vector<double> input_signal, std::vector<std::vector<double>> output_node, const int t_size, const int wash_out, const std::string name) {
	std::uniform_real_distribution<> rand_minus1toplus1(-1, 1);
	output_node[0][0] = 1.0;//�ύX����v�f
	for (int n = 1; n <= unit_size; n++) output_node[0][n] = rand_minus1toplus1(mt);

	std::ofstream outputfile("output_unit/" + name + ".txt");//output_unit �����I
	outputfile << "t,unit,input,output" << std::endl;

	const double e = 2.718;// 281828459045;
	double ��, d;
	d = 95 / (double)unit_size;//���� +1��������
	�� = log(1.0 + d);

	std::vector<double> input_sum_node(unit_size + 1, 0);    //�v�f��unit_size+1�A�S�Ă̗v�f�̒l0 �ŏ�����

	for (int t = 1; t <= t_size; t++) {
		for (int n = 1; n <= unit_size; n++) {
			J[t][n] = input_signal[t - 1] * input_signal_strength[n];
		}
	}

	for (int t = 1; t <= t_size; t++) {//t = 0��t = 1�ɕύX
		output_node[t][0] = output_node[t - 1][unit_size];
		for (int n = 1; n <= unit_size; n++) {
			output_node[t][n] = activation_function(output_node[t - 1][n], node_type[n], J[t][n]);//�����̈��������Ƒ����邩��
			//output_node[t][n] = activation_function(output_node[t - 1][n], node_type[n]);
			output_node[t][n] *= (1 - pow(e, -��));
			output_node[t][n] += pow(e, -��) * (output_node[t][n - 1]);
		}
		for (int n = 1; n <= unit_size; n++) {
			if (t >= wash_out && t < wash_out + 200)
				outputfile << t << "," << n << "," << input_sum_node[n] << "," << output_node[t + 1][n] << std::endl;
		}
	}
	outputfile.close();
}


bool reservoir_layer::is_echo_state_property(const std::vector<double>& input_signal) {
	auto output_node1 = std::vector<std::vector<double>>(wash_out + 2, std::vector<double>(unit_size + 1, 0));
	auto output_node2 = std::vector<std::vector<double>>(wash_out + 2, std::vector<double>(unit_size + 1, 0));

	reservoir_update(input_signal, output_node1, wash_out, 1);
	reservoir_update(input_signal, output_node2, wash_out, 2);

	double err_sum = 0.0;
	for (int t = wash_out - 99; t <= wash_out; t++) {
		for (int n = 1; n <= unit_size; n++) {
			const double err = (output_node1[t][n] - output_node2[t][n]);
			err_sum += err * err;
		}
	}
	
	// �m�[�h�����l�ɂ���ď�Ԃ��������Ȃ�Ȃ�΁AEchoStateProperty������
	double err_ave = err_sum / (unit_size * 10);
	//std::cout << err_ave << "\n";
	std::cerr << err_sum << std::endl;
	return err_ave <= 0.2;//������
}


double reservoir_layer::activation_function(const double x, const int type, const double J) {//�����̈��������Ƒ����邩��
//double reservoir_layer::activation_function(const double x, const int type) {
	if (type == LINEAR) {
		return std::max(-1000.0, std::min(1000.0, x));  //?
	}
	else if (type == NON_LINEAR) {
		//return nonlinear(x);
		return nonlinear(x, J, input_gain, feed_gain);
		//����������Ɓ@nonlinear(x, input_gain, feed_gain, pa, J);�Ȃ̂���
	}
	assert(type != LINEAR && type != NON_LINEAR);  //?
	return -1.0;
}
