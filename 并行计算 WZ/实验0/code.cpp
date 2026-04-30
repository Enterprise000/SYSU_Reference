#include<iostream>
#include<vector>
#include<chrono>
#include<cstdlib>
#include<ctime>
using namespace std;
using namespace chrono;

vector<vector<int>> CreateMatirx(int m, int n) {
	srand(time(0));
	vector<vector<int>> ans(m, vector<int>(n));
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			ans[i][j] = rand() % (2048 - 512 + 1) + 512;
		}
	}
	return ans;
}

vector<vector<int>> CreateMatirx1(int m, int n) {
	srand(time(0));
	vector<vector<int>> ans(m, vector<int>(n));
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			ans[i][j] = 0;
		}
	}
	return ans;
}

int main() {
	int m = 100;
	int n = 150;
	int k = 200;
	vector<vector<int>> A = CreateMatirx(m, n);
	vector<vector<int>> B = CreateMatirx(n, k);
	vector<vector<int>> C = CreateMatirx1(m, k);
	auto start = high_resolution_clock::now();
	for (int i = 0; i < m; i+=2) {
		for (int a = 0; a < n; a++) {
			for (int j= 0; j < k; j+=2) {
				C[i][j] = C[i][j] + A[i][a] * B[a][j];
				C[i + 1][j] += A[i + 1][a] * B[a][j];
				C[i][j + 1] += A[i][a] * B[a][j + 1];
				C[i + 1][j + 1] += A[i + 1][a] * B[a][j + 1];
			}
		}
	}
	auto end = high_resolution_clock::now();
	auto time = duration_cast<milliseconds>(end - start);
	cout << "runtime: " << time.count() << "milliseconds" << endl;
	return 0;
}
