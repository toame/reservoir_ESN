#include <cblas.h>
#include <vector>
class output_learning {
public:
    std::vector<std::vector<double>> A;
    std::vector<double> w;
    output_learning();

    //�A���ꎟ������Aw=b��A�𐶐�
    void generate_simultaneous_linear_equationsA(const std::vector<std::vector<double>>& output_node, const int wash_out, const int step, const int n_size);
   
    //�A���ꎟ������Aw=b��b�𐶐�
    void generate_simultaneous_linear_equationsb(std::vector<double>& b, const std::vector<std::vector<double>>& output_node, const std::vector<double>& yt_s, const int wash_out, const int step, const int n_size);


    // �A���ꎟ������Aw = b ��w�ɂ���ICCG�ŉ���
    // void Learning(std::vector<double>& w, const std::vector<std::vector<double>>& A, const std::vector<double>& b, const double lambda, const int n_size);

    // �s���S�R���X�L�[����
    int IncompleteCholeskyDecomp2(std::vector<std::vector<double>>& L, std::vector<double>& d, int n);
    

    // p_0 = (LDL^T)^-1 r_0 �̌v�Z
    void ICRes(const std::vector<std::vector<double>>& L, const std::vector<double>& d, const std::vector<double>& r, std::vector<double>& u, int n);

    // ���ς����߂�
    double dot(const std::vector<double> r1, const std::vector<double> r2, const int size);

    /*!
     * �s���S�R���X�L�[�����ɂ��O�����t�������z�@�ɂ��A�Ex=b������
     * @param[in] A n�~n���l�Ώ̍s��
     * @param[in] b �E�Ӄx�N�g��
     * @param[out] x ���ʃx�N�g��
     * @param[in] n �s��̑傫��
     * @param[inout] max_iter �ő唽����(�����I����,���ۂ̔�������Ԃ�)
     * @param[inout] eps ���e�덷(�����I����,���ۂ̌덷��Ԃ�)
     * @return 1:����,0:���s
     */
    int ICCGSolver(const std::vector<std::vector<double>>& L, const std::vector<double>& d, const std::vector<double>& b, std::vector<double>& x, int n, int& max_iter, double& eps);

    /*!
     * �������z�@�ɂ��A�Ex=b������
     * @param[in] A n�~n���l�Ώ̍s��
     * @param[in] b �E�Ӄx�N�g��
     * @param[out] x ���ʃx�N�g��
     * @param[in] n �s��̑傫��
     * @param[inout] max_iter �ő唽����(�����I����,���ۂ̔�������Ԃ�)
     * @param[inout] eps ���e�덷(�����I����,���ۂ̌덷��Ԃ�)
     * @return 1:����,0:���s
     */
    int CGSolver(const std::vector<std::vector<double>>& A, const std::vector<double>& b, std::vector<double>& x, int n, int& max_iter, double& eps);
};

inline double output_learning::dot(const std::vector<double> r1, const std::vector<double> r2, const int size) {
    return cblas_ddot(size, r1.data(), 1, r2.data(), 1);
}
