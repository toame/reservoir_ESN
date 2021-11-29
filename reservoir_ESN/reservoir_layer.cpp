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
	//std::vector<std::vector<double>> J; 
	J.resize(t_size + 1, std::vector<double>(unit_size + 1));
	//double �� = 2.0;//�B���@�����Őݒ�
	a.resize(6);
	b.resize(2);
	this->j = j;
	//this->j2 = j2;
}





// �����g�|���W�[�⌋���d�݂Ȃǂ�ݒ肷��.  �o�Q�lor6�l�̃����_���M�����p
void reservoir_layer::generate_reservoir() {
	 
	//std::uniform_real_distribution<> rand_minus1toplus1(-1, 1);//�����_������
	//std::uniform_int_distribution<> rand_minus2toplus2(-2, 2);//int������0��1
	//std::uniform_int_distribution<> rand_minus1orplus1(-10, 10);
	//std::uniform_int_distribution<> rand_0or5(-2, 3);
	a = { -1.0, -0.6, -0.2, 0.2, 0.6, 1.0 };
	b = {-1.0,1.0};

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
	/*for (int n = 1; n <= unit_size; n++) {

		if (b[rand() % b.size()] == 1) {
			node_type[n] = NON_LINEAR;
		}
		else
			node_type[n] = LINEAR;
	}*/

	// ���͑w�̌����d�݂����� �}�X�N�M���Ɠ��͂̋��݂������ňꏏ�ɂ��Ă���
	for (int n = 1; n <= unit_size; n++) {
		//input_signal_strength[n] = input_signal_factor * (double)(rand_minus1orplus1(mt) / 2.0);
		input_signal_strength[n] = input_signal_factor * a[rand() % a.size()];
		//input_signal_strength[n] = input_signal_factor * b[rand() % b.size()];///////////////////////////�ύX�v�f//////////////////////
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
	double exp(double x);
	output_node[0][0] = 1.0;//�ύX����v�f
	for (int n = 1; n <= unit_size; n++) output_node[0][n] = (double)rand_minus1toplus1(mt2);

	//std::vector<double> virtual_output_node(unit_size + 1, 0);


	//const double e = 2.7182818;// 2.718281828459045;
	double ��, d;
	//�� = (double) unit_size * 0.2;
	d = 33.0 / (double)unit_size;//�i�x�����ԁj��1�Ƃ��Ă��邪�_���ł�80�Ƃ��Ă���ꍇ��...�@�@////////////////////////�ύX�v�f/////////////////
	�� = log(1.0 + d);

	/*
	�� = 95 err_ave  0.1345

	11/11(��)
	50.0��narma�^�X�N��nmse0.020(���Ȃ݂ɃQ�C���͓���1.3�t�߁@�t�B�[�h��0.3�t�߂ɂ��Ă���)
	30�Ł@�@�@�@�@�@�@�@�@ 0.018
	31.4                   0.01645(pa = 2, d_alpha = 0.05; alpha_min = 0.80 )

	�C�Â�������
	�@input_gain��1�`1.3�Ō������\���ł₷����
	  feed_gain��0.3�t�߁@
	  pa�͒�߂̂ق����悳����(���2�̎����\������)
	*/


	//std::vector<double> input_sum_node(unit_size + 1, 0);    //�v�f��unit_size+1�A�S�Ă̗v�f�̒l0 �ŏ�����

	//�}�X�N�M�����������ŏI�I�ȓ��͐M��
	for (int t = 1; t <= t_size; t++) {
		for (int n = 1; n <= unit_size; n++) {
			J[t][n] = input_signal[t - 1] * input_signal_strength[n];
			//if (n <= 10) std::cerr << t << " " << n << " " << J[t][n] << " " << input_signal[t - 1] << " " << input_signal_strength[n] << std::endl;
		}
	}
	//�����l�ݒ�
	for (int n = 1; n <= unit_size; n++) {
		J[0][n] = input_signal_strength[n];
		output_node[0][n] = activation_function(output_node[0][n], 0.0, node_type[n], J[0][n]);
		//output_node[0][n] *= (1.0 - pow(e, -��));
		//output_node[0][n] *= d / (1.0 + d); /////////////////////////////////////////////////////�ύX�v�f//////////////////////
		output_node[0][n] *= (1.0 - exp(-��));
		output_node[0][n] += exp(-��) * (output_node[0][n - 1]);
		//output_node[0][n] += (1.0 / (1.0 + d)) * (output_node[0][n - 1]);/////////////////////////////////////////�ύX�v�f////////////////////
	}

	/*for (int t = 1; t <= t_size; t++) {//t = 0��t = 1�ɕύX
		output_node[t][0] = output_node[t - 1][unit_size];
		for (int n = 1; n <= unit_size; n++) {
			//output_node[t][n] = activation_function(output_node[t - 1][n], node_type[n], J[t][n]);
			output_node[t][n] = activation_function(output_node[t - 1][n], node_type[n]);
			output_node[t][n] *= (1 - pow(e, -��));
			output_node[t][n] += pow(e, -��) * (output_node[t][n - 1]);
		}
	}*/


		/*//�ʏ�̎��Ԓx���V�X�e���^���Ԕ��W��
	for (int t = 1; t <= t_size; t++) {//t = 0��t = 1�ɕύX
		output_node[t][0] = output_node[t - 1][unit_size];
		for (int n = 1; n <= unit_size; n++) {
			output_node[t][n] = activation_function(output_node[t - 1][n], 0.0, node_type[n], J[t][n]);//�����̈��������Ƒ����邩��
				//output_node[t][n] = activation_function(output_node[t - 1][n], node_type[n]);
				//if (n == 1) std::cerr << t << " " << output_node[t - 1][n] << " "<< output_node[t][n] << std::endl;
				//output_node[t][n] *= (1.0 - pow(e, -��));
			output_node[t][n] *= (1.0 - exp(-��));
				//output_node[t][n] *= (d / (1.0 + d));/////////////////////////////////////���\�悩����//////////////////
				//if (n == 1) std::cerr << t << " " << output_node[t][n] << std::endl;
				//output_node[t][n] += pow(e, -��) * (output_node[t][n - 1]);
			output_node[t][n] += exp(-��) * (output_node[t][n - 1]);
				//output_node[t][n] += (1.0 / (1.0 + d)) * (output_node[t][n - 1]);//////////////���\�悩����/////////////////////
				//if (n == 1) std::cerr << t << " " << output_node[t][n] << std::endl;
		}
	}*/




		//�񎟂̎��Ԓx���V�X�e���^���Ԕ��W��
	j = 80;
	for (int t = 1; t <= t_size; t++) {//t = 0��t = 1�ɕύX
		output_node[t][0] = output_node[t - 1][unit_size];
		for (int n = 1; n <= unit_size; n++) {
			if (t == 1) {
				output_node[t][n] = activation_function(output_node[t - 1][n], 0.0, node_type[n], J[t][n]);
				//if (n == 5) std::cerr << t << " " << output_node[t][n] << std::endl;
			}
			else if (t >= 2) {
				if (n >= j + 1) {
					output_node[t][n] = activation_function(output_node[t - 1][n], output_node[t - 1][n - j], node_type[n], J[t][n]);
					//if (n == 5) std::cerr << t << " " << output_node[t][n] << std::endl;
				}
				else {
					output_node[t][n] = activation_function(output_node[t - 1][n], output_node[t - 2][unit_size - j + n], node_type[n], J[t][n]);
					//if (n == 1) std::cerr << t << " " << output_node[t][n] << "_" << output_node[t - 2][unit_size - j + n] << std::endl;
				}

			}
			//if (n == 1) std::cerr << t << " " << output_node[t][n] << std::endl;

			//output_node[t][n] = activation_function(output_node[t - 1][n], node_type[n]);
			//if (n == 1) std::cerr << t << " " << output_node[t - 1][n] << " "<< output_node[t][n] << std::endl;
			//output_node[t][n] *= (1.0 - pow(e, -��));
			output_node[t][n] *= (1.0 - exp(-��));
			//output_node[t][n] *= (d / (1.0 + d));/////////////////////////////////////���\�悩����//////////////////
			//if (n == 1) std::cerr << t << " " << output_node[t][n] << std::endl;
			//output_node[t][n] += pow(e, -��) * (output_node[t][n - 1]);
			output_node[t][n] += exp(-��) * (output_node[t][n - 1]);
			//output_node[t][n] += (1.0 / (1.0 + d)) * (output_node[t][n - 1]);//////////////���\�悩����/////////////////////
			//if (n == 1) std::cerr << t << " " << output_node[t][n] << std::endl;
			//if (n == unit_size) std::cerr << t << " " << output_node[t][n] << std::endl;
		}
	}
}
/*	//�O���̎��Ԓx���V�X�e���^���Ԕ��W��
	j2 = 60;
	for (int t = 1; t <= t_size; t++) {
		output_node[t][0] = output_node[t - 1][unit_size];
		for (int n = 1; n <= unit_size; n++) {
			if (t == 1) {
				output_node[t][n] = activation_function2(output_node[t - 1][n], 0.0, 0.0, node_type[n], J[t][n]);
			}
			else if (t >= 2) {
				if (n >= j + 1) {
					output_node[t][n] = activation_function2(output_node[t - 1][n], output_node[t - 1][n - j], output_node[t - 1][n - j2], node_type[n], J[t][n]);
				}
				else if(n >= j2 + 1 && n < j + 1){
					output_node[t][n] = activation_function2(output_node[t - 1][n], output_node[t - 2][unit_size - j + n], output_node[t - 1][n - j2], node_type[n], J[t][n]);
				}
				else {
					output_node[t][n] = activation_function2(output_node[t - 1][n], output_node[t - 2][unit_size - j + n], output_node[t - 2][unit_size - j2 + n], node_type[n], J[t][n]);
				}
			}
			//output_node[t][n] *= (1.0 - exp(-��));
			output_node[t][n] *= (d / (1.0 + d));
			output_node[t][n] += (1.0 / (1.0 + d)) * (output_node[t][n - 1]);
			//output_node[t][n] += exp(-��) * (output_node[t][n - 1]);
		}
	}*/


void reservoir_layer::reservoir_update_show(const std::vector<double> input_signal, std::vector<std::vector<double>> output_node, const int t_size, const int wash_out, const std::string name) {
	std::uniform_real_distribution<> rand_minus1toplus1(-1, 1);
	output_node[0][0] = 1.0;//�ύX����v�f
	for (int n = 1; n <= unit_size; n++) output_node[0][n] = (double)rand_minus1toplus1(mt);

	std::ofstream outputfile("output_unit_STDE/" + name + ".txt");//output_unit �����I
	//outputfile << "t,unit,input,output" << std::endl;
	outputfile << "t,unit,output" << std::endl;

	//const double e = 2.7182818;// 281828459045;
	double ��, d;
	d = 17.0 / (double)unit_size;//���� +1���������@//////////////////////////////////////////�ύX�v�f//////////////////
	�� = log(1.0 + d);

	//std::vector<double> input_sum_node(unit_size + 1, 0);    //�v�f��unit_size+1�A�S�Ă̗v�f�̒l0 �ŏ�����

	for (int t = 1; t <= t_size; t++) {
		for (int n = 1; n <= unit_size; n++) {
			J[t][n] = input_signal[t - 1] * input_signal_strength[n];
		}
	}

	
	for (int n = 1; n <= unit_size; n++) {
		J[0][n] = input_signal_strength[n];
		output_node[0][n] = activation_function(output_node[0][n], 0.0, node_type[n], J[0][n]);
		//output_node[0][n] *= d / (1.0 + d); 
		output_node[0][n] *= (1.0 - exp(-��));
		//output_node[0][n] += (1.0 / (1.0 + d)) * (output_node[0][n - 1]);
		output_node[0][n] += exp(-��) * (output_node[0][n - 1]);
	}


	/*//�ʏ�̎��Ԓx���V�X�e���^���Ԕ��W��
	for (int t = 1; t <= t_size; t++) {//t = 0��t = 1�ɕύX
		output_node[t][0] = output_node[t - 1][unit_size];
		for (int n = 1; n <= unit_size; n++) {
			output_node[t][n] = activation_function(output_node[t - 1][n], 0.0, node_type[n], J[t][n]);//�����̈��������Ƒ����邩��
			//output_node[t][n] = activation_function(output_node[t - 1][n], node_type[n]);
			//output_node[t][n] *= (1.0 - pow(e, -��));
			output_node[t][n] *= (1.0 - exp(-��));
			//output_node[t][n] *= (d / (1.0 + d));///////////////////////////////////////////////////////////////
			//output_node[t][n] += pow(e, -��) * (output_node[t][n - 1]);
			output_node[t][n] += exp(-��) * (output_node[t][n - 1]);
			//output_node[t][n] += (1.0 / (1.0 + d)) * (output_node[t][n - 1]);////////////////////////////////////
		}
	}*/
		
	//�񎟂̎��Ԓx���V�X�e���^���Ԕ��W��
	const int j = 80;
	for (int t = 1; t <= t_size; t++) {//t = 0��t = 1�ɕύX
		output_node[t][0] = output_node[t - 1][unit_size];
		for (int n = 1; n <= unit_size; n++) {
			if (t == 1) 
				output_node[t][n] = activation_function(output_node[t - 1][n], 0.0, node_type[n], J[t][n]);
			else{
				if (n >= j + 1)
					output_node[t][n] = activation_function(output_node[t - 1][n], output_node[t - 1][n - j], node_type[n], J[t][n]);
				else
					output_node[t][n] = activation_function(output_node[t - 1][n], output_node[t - 2][unit_size - j + n], node_type[n], J[t][n]);
			}
				//�����̈��������Ƒ����邩��
				//output_node[t][n] = activation_function(output_node[t - 1][n], node_type[n]);
				//if (n == 1) std::cerr << t << " " << output_node[t - 1][n] << " "<< output_node[t][n] << std::endl;
				//output_node[t][n] *= (1.0 - pow(e, -��));
		    output_node[t][n] *= (1.0 - exp(-��));
			//output_node[t][n] *= (d / (1.0 + d));/////////////////////////////////////���\�悩����//////////////////
			//if (n == 1) std::cerr << t << " " << output_node[t][n] << std::endl;
			//output_node[t][n] += pow(e, -��) * (output_node[t][n - 1]);
			output_node[t][n] += exp(-��) * (output_node[t][n - 1]);
			//output_node[t][n] += (1.0 / (1.0 + d)) * (output_node[t][n - 1]);//////////////���\�悩����/////////////////////
			//if (n == 1) std::cerr << t << " " << output_node[t][n] << std::endl;
		}
		for (int n = 1; n <= unit_size; n++) {
			if (t >= wash_out && t < wash_out + 200)
			//outputfile << t << "," << n << "," << input_sum_node[n] << "," << output_node[t + 1][n] << std::endl;
			outputfile << t << "," << n << "," << output_node[t][n] << std::endl;
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
			//if (n == 1) {
			//	std::cerr << t << " " << err << std::endl;
			//}
		}
	}
	
	// �m�[�h�����l�ɂ���ď�Ԃ��������Ȃ�Ȃ�΁AEchoStateProperty������
	double err_ave = err_sum / (unit_size * 10);
	//std::cout << err_ave << "\n";
	//std::cerr << err_sum << std::endl;
	//std::cerr << input_signal_factor << " " << input_gain << " " << feed_gain << std::endl;
	/*if (unit_size < 40)
		return err_ave <= 0.2;
	else
		return err_ave <= 0.1;//������*/
	return err_ave <= 0.2;
}

double reservoir_layer::activation_function(const double x1,const double x2, const int type, const double J) {//�����̈��������Ƒ����邩��
//double reservoir_layer::activation_function(const double x, const int type) {
	double x;
	x = x1 + x2;

	if (type == LINEAR) {
		return std::max(-1000.0, std::min(1000.0, x)); 
	}
	else if (type == NON_LINEAR) {
		//return nonlinear(x);
		///double makkey(const double x, double J, double input_gain, double feed_gain) {//Mackey_Glass
		//return feed_gain * (x + input_gain * J) / (1.0 + pow(x + input_gain * J, 2.0));//�w���� = 2-------------------------
		//return feed_gain * sin(x + input_gain * J + 0.7) * sin(x + input_gain * J + 0.7);

		//return feed_gain * pow(sin(x + input_gain * J + 0.3), 2.0);//�r�c���f��  ��:�I�t�Z�b�g�ʑ�
		//return feed_gain * exp(-x) * sin(x + input_gain * J);//expsin �σp�����[�^�̒����K�v�Ȃ� ���\�ǂ�����
		//}

		return nonlinear(x, J, input_gain, feed_gain);
		//return nonlinear(x, input_gain, feed_gain, J);
	}
	assert(type != LINEAR && type != NON_LINEAR);  //?
	return -1.0;
}

/*
double reservoir_layer::activation_function2(const double x1, const double x2, const double x3, const int type, const double J) {//�����̈��������Ƒ����邩��
//double reservoir_layer::activation_function(const double x, const int type) {
	double x;
	x = x1 + x2 + x3;

	if (type == LINEAR) {
		return std::max(-1000.0, std::min(1000.0, x));
	}
	else if (type == NON_LINEAR) {
		//return nonlinear(x);
		///double makkey(const double x, double J, double input_gain, double feed_gain) {//Mackey_Glass
		//return feed_gain * (x + input_gain * J) / (1.0 + pow(x + input_gain * J, 2.0));//�w���� = 2-------------------------
		//return feed_gain * sin(x + input_gain * J + 0.7) * sin(x + input_gain * J + 0.7);

		//return feed_gain * pow(sin(x + input_gain * J + 0.3), 2.0);//�r�c���f��  ��:�I�t�Z�b�g�ʑ�
		//return feed_gain * exp(-x) * sin(x + input_gain * J);//expsin �σp�����[�^�̒����K�v�Ȃ� ���\�ǂ�����
		//}

		return nonlinear(x, J, input_gain, feed_gain);
		//return nonlinear(x, input_gain, feed_gain, J);
	}
	assert(type != LINEAR && type != NON_LINEAR);  //?
	return -1.0;
}
*/