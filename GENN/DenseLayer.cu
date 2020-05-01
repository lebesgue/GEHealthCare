#include "DenseLayer.cuh"
#include <stdexcept>

namespace cuda {


float __device__ get(Matrix A, int row, int col) {
	return A.data[row * A.w + col];
}


float __device__ get(Vector v, int pos) {
	return v.data[pos];
}


void __device__ set(Matrix A, int row, int col, float val) {
	A.data[row * A.w + col] = val;
}


void __device__ inc(Matrix A, int row, int col, float val) {
	A.data[row * A.w + col] += val;
}


void __global__ activate(float * x, int n, int activation) {
	int idx = threadIdx.x;
	if (idx >= n)
		return;

	if (activation == 0)
		x[idx] = 1.0f / (1.0f + expf(-x[idx]));
	if (activation == 1) {
		float sm = 0.0f;
		atomicAdd(&sm, expf(x[idx]));
		__syncthreads();
		x[idx] /= sm;
	}
}


void __global__ linear(DenseLayer layer) {
	extern __shared__ float sharedX[];

	int row = blockIdx.x;
	int col = threadIdx.x;

	if (row >= layer.W.h)
		return;
	if (col >= layer.W.w)
		return;

	sharedX[col] = get(layer.input, col);
	__syncthreads();

	__shared__ float val;
	if (col == 0)
		val = layer.b.data[row];
	__syncthreads();

	atomicAdd(&val, get(layer.W, row, col) * sharedX[col]);
	__syncthreads();

	if (col == 0)
		layer.output.data[row] = val;
}


__global__ void grad(DenseLayer layer) {
	int row = blockIdx.x;
	int col = threadIdx.x;

	if (row >= layer.gradW.h || col >= layer.gradW.w)
		return;

	__shared__ float gradAct;
	if (col == 0) {
		if (layer.activation == 0) {
			gradAct = layer.output.data[row] * (1 - layer.output.data[row]);
		}
		else {
			gradAct = 1.0f;
		}
	}
	__syncthreads();

	inc(layer.gradW, row, col, layer.input.data[col] * layer.dOutput.data[row] * gradAct);
	if (col == 0)
		layer.gradb.data[row] += gradAct * layer.dOutput.data[row];

}


__global__ void backPropagate(DenseLayer layer) {
	int row = threadIdx.x;
	int col = blockIdx.x;

	if (row >= layer.out || col >= layer.in)
		return;

	extern __shared__ float sharedX[];
	sharedX[row] = get(layer.W, row, col);
	if (layer.activation == 0) {
		sharedX[row + blockDim.x] = layer.output.data[row] * (1 - layer.output.data[row]);
	}
	else {
		sharedX[row + blockDim.x] = 1.0f;
	}
	__syncthreads();

	__shared__ float di;
	if (row == 0)
		di = 0.0f;
	__syncthreads();

	atomicAdd(&di, sharedX[row] * sharedX[row + blockDim.x] * layer.dOutput.data[row]);
	__syncthreads();

	if (row == 0)
		layer.dInput.data[col] = di;
}


__global__ void stepKernel(DenseLayer layer, float learningRate) {
	int row = blockIdx.x;
	int col = threadIdx.x;

	if (row >= layer.out || col >= layer.in)
		return;
	__syncthreads();

	inc(layer.W, row, col, -get(layer.gradW, row, col) * learningRate);
	if (col == 0)
		layer.b.data[row] -= layer.gradb.data[row] * learningRate;
}


inline cudaError_t checkCuda(cudaError_t result) {
	if (result != cudaSuccess) {
		throw std::runtime_error("CUDA Error");
	}
	return result;
}


void toGpu(float** dst, float** src, int s) {
	checkCuda(cudaMalloc(dst, s * sizeof(float)));
	checkCuda(cudaMemcpy(*dst, *src, s * sizeof(float), cudaMemcpyHostToDevice));
}


void fromGpu(float** dst, float** src, int s) {
	checkCuda(cudaMemcpy(*dst, *src, s * sizeof(float), cudaMemcpyDeviceToHost));
}


void DenseLayer::initLayer() {
	W.w = in;
	W.h = out;
	b.s = out;

	gradW.w = in;
	gradW.h = out;
	gradb.s = out;

	input.s = in;
	dInput.s = in;
	output.s = out;
	dOutput.s = out;

	checkCuda(cudaMalloc(&W.data, W.h * W.w * sizeof(float)));
	checkCuda(cudaMalloc(&gradW.data, gradW.h * gradW.w * sizeof(float)));
	checkCuda(cudaMalloc(&b.data, b.s * sizeof(float)));
	checkCuda(cudaMalloc(&gradb.data, gradb.s * sizeof(float)));
	checkCuda(cudaMemset(gradW.data, 0, gradW.h * gradW.w * sizeof(float)));
	checkCuda(cudaMemset(gradb.data, 0, gradb.s * sizeof(float)));
	
	checkCuda(cudaMalloc(&input.data, in * sizeof(float)));
	checkCuda(cudaMalloc(&dInput.data, in * sizeof(float)));
	checkCuda(cudaMalloc(&output.data, out * sizeof(float)));
	checkCuda(cudaMalloc(&dOutput.data, out * sizeof(float)));
}


void DenseLayer::destroyLayer() {
	checkCuda(cudaFree(&W.data));
	checkCuda(cudaFree(&gradW.data));
	checkCuda(cudaFree(&b.data));
	checkCuda(cudaFree(&gradb.data));
	checkCuda(cudaFree(&input.data));
	checkCuda(cudaFree(&dInput.data));
	checkCuda(cudaFree(&output.data));
	checkCuda(cudaFree(&dOutput.data));
}


void DenseLayer::forward() {
	int sharedMemSize = in * sizeof(float) * 2;
	int rows = out;
	int cols = in;

	linear<<<rows, cols, sharedMemSize>>>(*this);
	activate<<<1, rows>>>(output.data, out, activation);
	cudaDeviceSynchronize();
}


void DenseLayer::backward() {
	int sharedMemSize = out * sizeof(float) * 4;
	int rows = out;
	int cols = in;

	grad<<<rows, cols, sharedMemSize>>>(*this);
	backPropagate<<<cols, rows, sharedMemSize>>>(*this);
	cudaDeviceSynchronize();
}


void DenseLayer::step(float learningRate) {
	int rows = out;
	int cols = in;
	stepKernel<<<rows, cols>>>(*this, learningRate);
	cudaDeviceSynchronize();
}


void DenseLayer::zeroGrad() {
	checkCuda(cudaMemset(gradW.data, 0, gradW.h * gradW.w * sizeof(float)));
	checkCuda(cudaMemset(gradb.data, 0, gradb.s * sizeof(float)));
	cudaDeviceSynchronize();
}


void DenseLayer::initBackProp(int label) {
	if (!isOutput)
		return;

	float* o = (float*)malloc(out * sizeof(float));
	fromGpu(&o, &output.data, out);

	float sum = 0.0;
	for (int i = 0; i < out; i++) {
		o[i] = expf(o[i]);
		sum += o[i];
	}

	for (int i = 0; i < out; i++) {
		o[i] /= sum;
		o[i] -= i == label ? 1.0f : 0.0f;
	}
	toGpu(&dOutput.data, &o, out);
}


int DenseLayer::argmax() {
	float* o = (float*)malloc(out * sizeof(float));
	fromGpu(&o, &output.data, out);
	float m = o[0];
	int im = 0;
	for (int i = 1; i < out; i++) {
		if (o[i] > m) {
			im = i;
			m = o[i];
		}
	}
	return im;
}


float DenseLayer::loss(int label) {
	float denom = 0.0f;
	float * sm = (float *)malloc(out * sizeof(float));
	fromGpu(&sm, &output.data, out);
	for (int i = 0; i < out; i++) {
		sm[i] = expf(sm[i]);
		denom += sm[i];
	}
	return -logf(sm[label] / denom);
}


}
