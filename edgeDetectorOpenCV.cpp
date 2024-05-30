#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

using namespace cv;

void sobelEdgeDetection(const Mat& src, Mat& dst) {
    Mat gray, grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    Sobel(gray, grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
    convertScaleAbs(grad_x, abs_grad_x);
    Sobel(gray, grad_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT);
    convertScaleAbs(grad_y, abs_grad_y);
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst);
}

void prewittEdgeDetection(const Mat& src, Mat& dst) {
    Mat gray, grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    Mat kernel_x = (Mat_<float>(3, 3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);
    Mat kernel_y = (Mat_<float>(3, 3) << -1, -1, -1, 0, 0, 0, 1, 1, 1);

    filter2D(gray, grad_x, CV_16S, kernel_x
    );
    convertScaleAbs(grad_x, abs_grad_x);
    filter2D(gray, grad_y, CV_16S, kernel_y);
    convertScaleAbs(grad_y, abs_grad_y);
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst);
}

void scharrEdgeDetection(const Mat& src, Mat& dst) {
    Mat gray, grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;

    cvtColor(src, gray, COLOR_BGR2GRAY);
    Scharr(gray, grad_x, CV_16S, 1, 0, 1, 0, BORDER_DEFAULT);
    convertScaleAbs(grad_x, abs_grad_x);
    Scharr(gray, grad_y, CV_16S, 0, 1, 1, 0, BORDER_DEFAULT);
    convertScaleAbs(grad_y, abs_grad_y);
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst);
}

void morphologicalEdgeDetection(const Mat& src, Mat& dst) {
    Mat gray, morph_grad;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(gray, morph_grad, MORPH_GRADIENT, element);
    threshold(morph_grad, dst, 0, 255, THRESH_BINARY | THRESH_OTSU);
}

void laplacianEdgeDetection(const Mat& src, Mat& dst) {
    Mat gray, laplacian, abs_laplacian;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    Laplacian(gray, laplacian, CV_16S, 3, 1, 0, BORDER_DEFAULT);
    convertScaleAbs(laplacian, abs_laplacian);
    abs_laplacian.copyTo(dst);
}

int main() {
    std::ofstream resultFile("average_times_opencv.txt");
    if (!resultFile.is_open()) {
        std::cerr << "Error: Unable to open result file!" << std::endl;
        return -1;
    }

    for (int i = 1; i <= 4; ++i) {
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
            sobelEdgeDetection(src, sobel_result);
            prewittEdgeDetection(src, prewitt_result);
            scharrEdgeDetection(src, scharr_result);
            morphologicalEdgeDetection(src, morph_result);
            laplacianEdgeDetection(src, laplacian_result);
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
