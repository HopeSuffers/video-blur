#ifndef VIDEO_PROCESSOR_HPP
#define VIDEO_PROCESSOR_HPP

#include <opencv2/opencv.hpp>
#include <string>

class VideoProcessor {
public:
    VideoProcessor(const std::string& inputPath, const std::string& outputPath);

    // Single-threaded method
    void processSingleThreaded();

    // Multi-threaded method
    void processMultiThreaded(unsigned int numThreads);

    void processCUDA();  // new method

private:
    cv::VideoCapture cap;
    cv::VideoWriter writer;

    bool isOpened;
    int frameWidth;
    int frameHeight;
    double fps;

    void blurFrameSingle(const cv::Mat& src, cv::Mat& dst);
    void blurFrameMulti(const cv::Mat& src, cv::Mat& dst, unsigned int numThreads);
    void blurFrameCUDA(const cv::Mat& src, cv::Mat& dst);

    // Optionally, you can separate filter logic into its own method
    void applyFilter(const cv::Mat& src, cv::Mat& dst, int startRow, int endRow);
};

#endif
