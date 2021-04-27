#pragma once

#include <cblas.h>

#include <algorithm>
#include <cassert>
#include <iterator>
#include <random>
#include <vector>
#include <iostream>

#include "output_learning.h"

output_learning::output_learning() {}

//連立一次方程式Aw=bのAを生成
void output_learning::generate_simultaneous_linear_equationsA(const std::vector<std::vector<double>>& output_node, const int wash_out, const int step, const int n_size) {
	std::vector<std::vector<double>> C(n_size + 1, std::vector<double>(n_size + 1));
	A.resize(n_size + 1, std::vector<double>(n_size + 1));
	int count = step - wash_out;
	std::vector<double> sub_output_node(count * (n_size + 1));
	std::vector<double> B((n_size + 1) * (n_size + 1), 0.0);
	for (int t = wash_out + 1; t < step; t++) {
		for (int n1 = 0; n1 <= n_size; n1++) {
			sub_output_node[(t - wash_out - 1) * (n_size + 1) + n1] = output_node[t + 1][n1];
		}
	}

	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n_size + 1, n_size + 1, count, 1.0 / count, sub_output_node.data(), n_size + 1, sub_output_node.data(), n_size + 1, 0.0, B.data(), n_size + 1);
	for (int n1 = 0; n1 <= n_size; n1++) {
		for (int n2 = 0; n2 <= n_size; n2++) {
			A[n1][n2] = B[n1 * (n_size + 1) + n2];
		}
	}
}
//連立一次方程式Aw=bのbを生成
void output_learning::generate_simultaneous_linear_equationsb(const std::vector<std::vector<double>>& output_node,
	const std::vector<double>& yt_s, const int wash_out, const int step, const int n_size) {
	b.resize(n_size + 1);
	for (int n1 = 0; n1 <= n_size; n1++) {
		b[n1] = 0.0;
	}
	int count = 0;
	for (int t = wash_out + 1; t < step; t++) {
		count++;
		for (int n1 = 0; n1 <= n_size; n1++) {
			b[n1] += output_node[t + 1][n1] * yt_s[t];
		}
	}
	for (int n1 = 0; n1 <= n_size; n1++) {
		b[n1] /= count;
	}
}

// 連立一次方程式Aw = b をwについてICCGで解く
// void Learning(std::vector<double>& w, const std::vector<std::vector<double>>& A, const std::vector<double>& b, const double lambda,
//               const int n_size) {
//     std::vector<std::vector<double>> Atilde(n_size + 1, std::vector<double>(n_size + 1));
//     for(int n1 = 0; n1 <= n_size; n1++) {
//         for(int n2 = 0; n2 <= n_size; n2++) {
//             Atilde[n1][n2] = A[n1][n2] + (n1 == n2) * lambda;
//         }
//     }
//     int max_iter = n_size + 1;
//     double eps = 1e-12;
//     ICCGSolver(Atilde, b, w, Atilde.size(), max_iter, eps);
// }

// 不完全コレスキー分解
int output_learning::IncompleteCholeskyDecomp2(int n) {
	if (n <= 0) return 0;
	L.resize(n, std::vector<double>(n));
	d.resize(n);
	L[0][0] = A[0][0];
	d[0] = 1.0 / L[0][0];

	for (int i = 1; i < n; ++i) {
		for (int j = 0; j <= i; ++j) {
			if (fabs(A[i][j]) < 1.0e-10) continue;

			double lld = A[i][j];
			for (int k = 0; k < j; ++k) {
				lld -= L[i][k] * L[j][k] * d[k];
			}
			L[i][j] = lld;
		}

		d[i] = 1.0 / L[i][i];
	}

	return 1;
}



/*!
 * 不完全コレスキー分解による前処理付共役勾配法によりA・x=bを解く
 * @param[in] A n×n正値対称行列
 * @param[in] b 右辺ベクトル
 * @param[out] x 結果ベクトル
 * @param[in] n 行列の大きさ
 * @param[inout] max_iter 最大反復数(反復終了後,実際の反復数を返す)
 * @param[inout] eps 許容誤差(反復終了後,実際の誤差を返す)
 * @return 1:成功,0:失敗
 */
int output_learning::ICCGSolver(int n, int& max_iter, double& eps) {
	if (n <= 0) return 0;

	std::vector<double> r(n), p(n), y(n), r2(n);
	w.assign(n, 0.0);

	// 第0近似解に対する残差の計算
	for (int i = 0; i < n; ++i) {
		double ax = 0.0;
		for (int j = 0; j < n; ++j) {
			ax += A[i][j] * w[j];
		}
		r[i] = b[i] - ax;
	}

	// p_0 = (LDL^T)^-1 r_0 の計算
	ICRes(r, p, n);

	double rr0 = dot(r, p, n), rr1;
	double alpha, beta;

	double e = 0.0;
	int k;
	std::vector<double> B(n * n);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			B[i * n + j] = A[i][j];
		}
	}
	for (k = 0; k < max_iter; ++k) {
		// y = AP の計算
		//for (int i = 0; i < n; ++i) {
		//	y[i] = dot(A[i], p, n);
		//}
		cblas_dgemv(CblasRowMajor, CblasNoTrans, n, n, 1.0, B.data(), n, p.data(), 1, 0.0, y.data(), 1);

		// alpha = r*r/(P*AP)の計算
		alpha = rr0 / dot(p, y, n);

		// 解x、残差rの更新
		for (int i = 0; i < n; ++i) {
			w[i] += alpha * p[i];
			r[i] -= alpha * y[i];
		}

		// (r*r)_(k+1)の計算
		ICRes(r, r2, n);
		rr1 = dot(r, r2, n);

		// 収束判定 (||r||<=eps)
		e = sqrt(rr1);
		if (e < eps) {
			k++;
			break;
		}

		// βの計算とPの更新
		beta = rr1 / rr0;
		for (int i = 0; i < n; ++i) {
			p[i] = r2[i] + beta * p[i];
		}

		// (r*r)_(k+1)を次のステップのために確保しておく
		rr0 = rr1;
	}

	max_iter = k;
	eps = e;

	return 1;
}

/*!
 * 共役勾配法によりA・x=bを解く
 * @param[in] A n×n正値対称行列
 * @param[in] b 右辺ベクトル
 * @param[out] x 結果ベクトル
 * @param[in] n 行列の大きさ
 * @param[inout] max_iter 最大反復数(反復終了後,実際の反復数を返す)
 * @param[inout] eps 許容誤差(反復終了後,実際の誤差を返す)
 * @return 1:成功,0:失敗
 */
int output_learning::CGSolver(const std::vector<std::vector<double>>& A, const std::vector<double>& b, std::vector<double>& x, int n, int& max_iter, double& eps) {
	if (n <= 0) return 0;

	std::vector<double> r(n), p(n), y(n);
	x.assign(n, 0.0);

	// 第0近似解に対する残差の計算
	for (int i = 0; i < n; ++i) {
		double ax = 0.0;
		for (int j = 0; j < n; ++j) {
			ax += A[i][j] * x[j];
		}
		r[i] = b[i] - ax;
		p[i] = r[i];
	}

	double rr0 = dot(r, r, n), rr1;
	double alpha, beta;

	double e = 0.0;
	int k;
	for (k = 0; k < max_iter; ++k) {
		// y = AP の計算
		for (int i = 0; i < n; ++i) {
			y[i] = dot(A[i], p, n);
		}

		// alpha = r*r/(P*AP)の計算
		alpha = rr0 / dot(p, y, n);

		// 解x、残差rの更新
		for (int i = 0; i < n; ++i) {
			x[i] += alpha * p[i];
			r[i] -= alpha * y[i];
		}

		// (r*r)_(k+1)の計算
		rr1 = dot(r, r, n);

		// 収束判定 (||r||<=eps)
		e = sqrt(rr1);
		if (e < eps) {
			k++;
			break;
		}

		// βの計算とPの更新
		beta = rr1 / rr0;
		for (int i = 0; i < n; ++i) {
			p[i] = r[i] + beta * p[i];
		}

		// (r*r)_(k+1)を次のステップのために確保しておく
		rr0 = rr1;
	}

	max_iter = k + 1;
	eps = e;

	return 1;
}

