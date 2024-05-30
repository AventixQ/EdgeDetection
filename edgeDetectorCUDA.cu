#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>
#include <fstream>

using namespace cv;

__global__ void sobelEdgeDetectionKernel(const uchar* src, uchar* dst, int rows, int cols) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < cols && y < rows) {
        int gx = 0, gy = 0;

        if (x > 0 && x < cols - 1 && y > 0 && y < rows - 1) {
            gx = src[(y - 1) * cols + (x + 1)] - src[(y - 1) * cols + (x - 1)] +
                2 * src[y * cols + (x + 1)] - 2 * src[y * cols + (x - 1)] +
                src[(y + 1) * cols + (x + 1)] - src[(y + 1) * cols + (x - 1)];

            gy = src[(y + 1) * cols + (x - 1)] + 2 * src[(y + 1) * cols + x] + src[(y + 1) * cols + (x + 1)] -
                src[(y - 1) * cols + (x - 1)] - 2 * src[(y - 1) * cols + x] - src[(y - 1) * cols + (x + 1)];
        }

        int gradient = abs(gx) + abs(gy);
        dst[y * cols + x] = gradient > 255 ? 255 : gradient;
    }
}

void sobelEdgeDetectionCUDA(const Mat& src, Mat& dst) {
    int rows = src.rows;
    int cols = src.cols;

    size_t srcBytes = src.step * rows;
    size_t dstBytes = dst.step * rows;

    uchar* d_src, * d_dst;
    cudaMalloc(&d_src, srcBytes);
    cudaMalloc(&d_dst, dstBytes);

    cudaMemcpy(d_src, src.ptr(), srcBytes, cudaMemcpyHostToDevice);

    dim3 blockSize(32, 32);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

    sobelEdgeDetectionKernel << <gridSize, blockSize >> > (d_src, d_dst, rows, cols);

    cudaMemcpy(dst.ptr(), d_dst, dstBytes, cudaMemcpyDeviceToHost);

    cudaFree(d_src);
    cudaFree(d_dst);
}

__global__ void prewittEdgeDetectionKernel(const uchar* src, uchar* dst, int rows, int cols) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < cols && y < rows) {
        int gx = 0, gy = 0;

        if (x > 0 && x < cols - 1 && y > 0 && y < rows - 1) {
            gx = src[(y - 1) * cols + (x + 1)] - src[(y - 1) * cols + (x - 1)] +
                src[y * cols + (x + 1)] - src[y * cols + (x - 1)] +
                src[(y + 1) * cols + (x + 1)] - src[(y + 1) * cols + (x - 1)];

            gy = src[(y + 1) * cols + (x - 1)] + src[(y + 1) * cols + x] + src[(y + 1) * cols + (x + 1)] -
                src[(y - 1) * cols + (x - 1)] - src[(y - 1) * cols + x] - src[(y - 1) * cols + (x + 1)];
        }

        int gradient = abs(gx) + abs(gy);
        dst[y * cols + x] = gradient > 255 ? 255 : gradient;
    }
}

void prewittEdgeDetectionCUDA(const Mat& src, Mat& dst) {
    int rows = src.rows;
    int cols = src.cols;

    size_t srcBytes = src.step * rows;
    size_t dstBytes = dst.step * rows;

    uchar* d_src, * d_dst;
    cudaMalloc(&d_src, srcBytes);
    cudaMalloc(&d_dst, dstBytes);

    cudaMemcpy(d_src, src.ptr(), srcBytes, cudaMemcpyHostToDevice);

    dim3 blockSize(32, 32);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

    prewittEdgeDetectionKernel << <gridSize, blockSize >> > (d_src, d_dst, rows, cols);

    cudaMemcpy(dst.ptr(), d_dst, dstBytes, cudaMemcpyDeviceToHost);

    cudaFree(d_src);
    cudaFree(d_dst);
}

__global__ void scharrEdgeDetectionKernel(const uchar* src, uchar* dst, int rows, int cols) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < cols && y < rows) {
        int gx = 0, gy = 0;

        if (x > 0 && x < cols - 1 && y > 0 && y < rows - 1) {
            gx = -3 * src[(y - 1) * cols + (x - 1)] - 10 * src[y * cols + (x - 1)] - 3 * src[(y + 1) * cols + (x - 1)] +
                3 * src[(y - 1) * cols + (x + 1)] + 10 * src[y * cols + (x + 1)] + 3 * src[(y + 1) * cols + (x + 1)];

            gy = -3 * src[(y - 1) * cols + (x - 1)] - 10 * src[(y - 1) * cols + x] - 3 * src[(y - 1) * cols + (x + 1)] +
                3 * src[(y + 1) * cols + (x - 1)] + 10 * src[(y + 1) * cols + x] + 3 * src[(y + 1) * cols + (x + 1)];
        }

        int gradient = abs(gx) + abs(gy);
        dst[y * cols + x] = gradient > 255 ? 255 : gradient;
    }
}

void scharrEdgeDetectionCUDA(const Mat& src, Mat& dst) {
    int rows = src.rows;
    int cols = src.cols;

    size_t srcBytes = src.step * rows;
    size_t dstBytes = dst.step * rows;

    uchar* d_src, * d_dst;
    cudaMalloc(&d_src, srcBytes);
    cudaMalloc(&d_dst, dstBytes);

    cudaMemcpy(d_src, src.ptr(), srcBytes, cudaMemcpyHostToDevice);

    dim3 blockSize(32, 32);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

    scharrEdgeDetectionKernel << <gridSize, blockSize >> > (d_src, d_dst, rows, cols);

    cudaMemcpy(dst.ptr(), d_dst, dstBytes, cudaMemcpyDeviceToHost);

    cudaFree(d_src);
    cudaFree(d_dst);
}

__global__ void morphologicalEdgeDetectionKernel(const uchar* src, uchar* dst, int rows, int cols) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < cols && y < rows) {
        int gx = 0, gy = 0;

        if (x > 0 && x < cols - 1 && y > 0 && y < rows - 1) {
            int center = src[y * cols + x];
            int top = src[(y - 1) * cols + x];
            int bottom = src[(y + 1) * cols + x];
            int left = src[y * cols + (x - 1)];
            int right = src[y * cols + (x + 1)];

            gx = abs(right - left);
            gy = abs(bottom - top);
        }

        int gradient = gx + gy;
        dst[y * cols + x] = gradient > 255 ? 255 : gradient;
    }
}

void morphologicalEdgeDetectionCUDA(const Mat& src, Mat& dst) {
    int rows = src.rows;
    int cols = src.cols;

    size_t srcBytes = src.step * rows;
    size_t dstBytes = dst.step * rows;

    uchar* d_src, * d_dst;
    cudaMalloc(&d_src, srcBytes);
    cudaMalloc(&d_dst, dstBytes);

    cudaMemcpy(d_src, src.ptr(), srcBytes, cudaMemcpyHostToDevice);

    dim3 blockSize(32, 32);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

    morphologicalEdgeDetectionKernel << <gridSize, blockSize >> > (d_src, d_dst, rows, cols);

    cudaMemcpy(dst.ptr(), d_dst, dstBytes, cudaMemcpyDeviceToHost);

    cudaFree(d_src);
    cudaFree(d_dst);
}

__global__ void laplacianEdgeDetectionKernel(const uchar* src, uchar* dst, int rows, int cols) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < cols && y < rows) {
        int gx = 0, gy = 0;

        if (x > 0 && x < cols - 1 && y > 0 && y < rows - 1) {
            gx = -src[(y - 1) * cols + x] - src[y * cols + (x - 1)] + 4 * src[y * cols + x] - src[y * cols + (x + 1)] - src[(y + 1) * cols + x];
            gy = -src[(y - 1) * cols + x] - src[y * cols + (x - 1)] + 4 * src[y * cols + x] - src[y * cols + (x + 1)] - src[(y + 1) * cols + x];
        }

        int gradient = abs(gx) + abs(gy);
        dst[y * cols + x] = gradient > 255 ? 255 : gradient;
    }
}

void laplacianEdgeDetectionCUDA(const Mat& src, Mat& dst) {
    int rows = src.rows;
    int cols = src.cols;

    size_t srcBytes = src.step * rows;
    size_t dstBytes = dst.step * rows;

    uchar* d_src, * d_dst;
    cudaMalloc(&d_src, srcBytes);
    cudaMalloc(&d_dst, dstBytes);

    cudaMemcpy(d_src, src.ptr(), srcBytes, cudaMemcpyHostToDevice);

    dim3 blockSize(32, 32);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

    laplacianEdgeDetectionKernel << <gridSize, blockSize >> > (d_src, d_dst, rows, cols);

    cudaMemcpy(dst.ptr(), d_dst, dstBytes, cudaMemcpyDeviceToHost);

    cudaFree(d_src);
    cudaFree(d_dst);
}


int main() {
    std::ofstream resultFile("average_times.txt");
    if (!resultFile.is_open()) {
        std::cerr << "Error: Unable to open result file!" << std::endl;
        return -1;
    }

    for (int i = 1; i <= 15; ++i) {
        std::string filename = "input/photo" + std::to_string(i) + ".jpg";
        Mat src = imread(filename);
        if (src.empty()) {
            std::cerr << "Error: Unable to load image " << filename << "!" << std::endl;
            continue;
        }

        Mat gray;
        cvtColor(src, gray, COLOR_BGR2GRAY);
        Mat sobel_result(gray.size(), CV_8UC1);
        Mat prewitt_result(gray.size(), CV_8UC1);
        Mat scharr_result(gray.size(), CV_8UC1);
        Mat morph_result(gray.size(), CV_8UC1);
        Mat laplacian_result(gray.size(), CV_8UC1);

        double total_time = 0.0;
        for (int j = 0; j < 10; ++j) {


            TickMeter tm;
            tm.start();
            sobelEdgeDetectionCUDA(gray, sobel_result);
            prewittEdgeDetectionCUDA(gray, prewitt_result);
            scharrEdgeDetectionCUDA(gray, scharr_result);
            morphologicalEdgeDetectionCUDA(gray, morph_result);
            laplacianEdgeDetectionCUDA(gray, laplacian_result);

            tm.stop();

            total_time += tm.getTimeMilli();
        }

        double average_time = total_time / 10.0;
        resultFile << "photo" << i << ";" << average_time << std::endl;

        imwrite("results/sobel_result_" + std::to_string(i) + ".jpg", sobel_result);
        imwrite("results/prewitt_result_" + std::to_string(i) + ".jpg", prewitt_result);
        imwrite("results/scharr_result_" + std::to_string(i) + ".jpg", scharr_result);
        imwrite("results/morph_result_" + std::to_string(i) + ".jpg", morph_result);
        imwrite("results/laplacian_result_" + std::to_string(i) + ".jpg", laplacian_result);
    }

    resultFile.close();
    return 0;
}
