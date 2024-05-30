#include <iostream>
#include <omp.h>
#include <fstream>

using namespace cv;

void sobelEdgeDetection(const Mat& gray, Mat& dst) {
    for (int y = 1; y < gray.rows - 1; ++y) {
        for (int x = 1; x < gray.cols - 1; ++x) {
            int gx = -gray.at<uchar>(y - 1, x - 1) - 2 * gray.at<uchar>(y, x - 1) - gray.at<uchar>(y + 1, x - 1)
                + gray.at<uchar>(y - 1, x + 1) + 2 * gray.at<uchar>(y, x + 1) + gray.at<uchar>(y + 1, x + 1);

            int gy = -gray.at<uchar>(y - 1, x - 1) - 2 * gray.at<uchar>(y - 1, x) - gray.at<uchar>(y - 1, x + 1)
                + gray.at<uchar>(y + 1, x - 1) + 2 * gray.at<uchar>(y + 1, x) + gray.at<uchar>(y + 1, x + 1);

            int gradient = abs(gx) + abs(gy);
            dst.at<uchar>(y, x) = gradient > 255 ? 255 : gradient;
        }
    }
}

void prewittEdgeDetection(const Mat& gray, Mat& dst) {
    for (int y = 1; y < gray.rows - 1; ++y) {
        for (int x = 1; x < gray.cols - 1; ++x) {
            int gx = -gray.at<uchar>(y - 1, x - 1) - gray.at<uchar>(y, x - 1) - gray.at<uchar>(y + 1, x - 1)
                + gray.at<uchar>(y - 1, x + 1) + gray.at<uchar>(y, x + 1) + gray.at<uchar>(y + 1, x + 1);

            int gy = -gray.at<uchar>(y - 1, x - 1) - gray.at<uchar>(y - 1, x) - gray.at<uchar>(y - 1, x + 1)
                + gray.at<uchar>(y + 1, x - 1) + gray.at<uchar>(y + 1, x) + gray.at<uchar>(y + 1, x + 1);

            int gradient = abs(gx) + abs(gy);
            dst.at<uchar>(y, x) = gradient > 255 ? 255 : gradient;
        }
    }
}

void scharrEdgeDetection(const Mat& gray, Mat& dst) {
    for (int y = 1; y < gray.rows - 1; ++y) {
        for (int x = 1; x < gray.cols - 1; ++x) {
            int gx = -3 * gray.at<uchar>(y - 1, x - 1) - 10 * gray.at<uchar>(y, x - 1) - 3 * gray.at<uchar>(y + 1, x - 1)
                + 3 * gray.at<uchar>(y - 1, x + 1) + 10 * gray.at<uchar>(y, x + 1) + 3 * gray.at<uchar>(y + 1, x + 1);

            int gy = -3 * gray.at<uchar>(y - 1, x - 1) - 10 * gray.at<uchar>(y - 1, x) - 3 * gray.at<uchar>(y - 1, x + 1)
                + 3 * gray.at<uchar>(y + 1, x - 1) + 10 * gray.at<uchar>(y + 1, x) + 3 * gray.at<uchar>(y + 1, x + 1);

            int gradient = abs(gx) + abs(gy);
            dst.at<uchar>(y, x) = gradient > 255 ? 255 : gradient;
        }
    }
}

void morphologicalEdgeDetection(const Mat& gray, Mat& dst) {
    for (int y = 1; y < gray.rows - 1; ++y) {
        for (int x = 1; x < gray.cols - 1; ++x) {
            int gx = abs(gray.at<uchar>(y, x + 1) - gray.at<uchar>(y, x - 1));
            int gy = abs(gray.at<uchar>(y + 1, x) - gray.at<uchar>(y - 1, x));
            int gradient = gx + gy;
            dst.at<uchar>(y, x) = gradient > 255 ? 255 : gradient;
        }
    }
}

void laplacianEdgeDetection(const Mat& gray, Mat& dst) {
    for (int y = 1; y < gray.rows - 1; ++y) {
        for (int x = 1; x < gray.cols - 1; ++x) {
            int gx = -gray.at<uchar>(y - 1, x) - gray.at<uchar>(y, x - 1) + 4 * gray.at<uchar>(y, x) - gray.at<uchar>(y, x + 1) - gray.at<uchar>(y + 1, x);
            int gy = gx; // Laplacian is the same in both directions.
            int gradient = abs(gx) + abs(gy);
            dst.at<uchar>(y, x) = gradient > 255 ? 255 : gradient;
        }
    }
}

int main() {
    std::ofstream resultFile("average_times_normal.txt");
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
            sobelEdgeDetection(gray, sobel_result);
            prewittEdgeDetection(gray, prewitt_result);
            scharrEdgeDetection(gray, scharr_result);
            morphologicalEdgeDetection(gray, morph_result);
            laplacianEdgeDetection(gray, laplacian_result);
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