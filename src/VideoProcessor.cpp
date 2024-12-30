#include "VideoProcessor.hpp"
#include "GpuBlur.hpp"
#include <thread>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

VideoProcessor::VideoProcessor(const std::string& inputPath, const std::string& outputPath)
{
    cap.open(inputPath);
    if (!cap.isOpened()) {
        std::cerr << "Could not open input video file: " << inputPath << std::endl;
        isOpened = false;
        return;
    }
    isOpened = true;

    frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    fps = cap.get(cv::CAP_PROP_FPS);

    // Define codec and create VideoWriter object
    writer.open(
        outputPath,
        cv::VideoWriter::fourcc('M','J','P','G'),
        fps,
        cv::Size(frameWidth, frameHeight)
    );
}

void VideoProcessor::processSingleThreaded()
{
    if (!isOpened) return;

    cv::Mat frame, output;
    while (true) {
        if (!cap.read(frame) || frame.empty()) {
            break;
        }

        // Single-threaded blur
        blurFrameSingle(frame, output);

        // Write or display
        writer.write(output);
        // cv::imshow("Single-Threaded", output);
        // if (cv::waitKey(1) == 27) { break; } // ESC to exit
    }
    cap.release();
    writer.release();
}

void VideoProcessor::processMultiThreaded(unsigned int numThreads)
{
    if (!isOpened) return;

    cv::Mat frame, output;
    while (true) {
        if (!cap.read(frame) || frame.empty()) {
            break;
        }

        // Multi-threaded blur
        blurFrameMulti(frame, output, numThreads);

        writer.write(output);
        // cv::imshow("Multi-Threaded", output);
        // if (cv::waitKey(1) == 27) { break; }
    }
    cap.release();
    writer.release();
}

void VideoProcessor::blurFrameSingle(const cv::Mat& src, cv::Mat& dst)
{
    // For demonstration, let's use cv::blur (box filter). You can use Gaussian if you prefer.
    cv::GaussianBlur(src, dst, cv::Size(15, 15), 0);
}

// This function spawns threads to process different regions of the frame
void VideoProcessor::blurFrameMulti(const cv::Mat& src, cv::Mat& dst, unsigned int numThreads)
{
    dst = src.clone(); // or create new Mat of the same size

    // Each thread will process a portion of the rows
    std::vector<std::thread> threads;
    int rowsPerThread = frameHeight / numThreads;

    for (unsigned int t = 0; t < numThreads; ++t) {
        int startRow = t * rowsPerThread;
        int endRow = (t == numThreads - 1) ? frameHeight : (t + 1) * rowsPerThread;

        threads.emplace_back([this, &src, &dst, startRow, endRow]() {
            // Apply the filter to the sub-region
            applyFilter(src, dst, startRow, endRow);
        });
    }

    for (auto &thr : threads) {
        thr.join();
    }
}

void VideoProcessor::applyFilter(const cv::Mat& src, cv::Mat& dst, int startRow, int endRow)
{
    // For each row, apply the same blur logic. This is a simplified demonstration:
    cv::Mat subSrc = src(cv::Range(startRow, endRow), cv::Range::all());
    cv::Mat subDst = dst(cv::Range(startRow, endRow), cv::Range::all());

    cv::blur(subSrc, subDst, cv::Size(5,5));
}

// Forward declaration of the CUDA function we wrote
extern void runBlurKernel(const unsigned char* inData, unsigned char* outData,
                          int width, int height, int channels);


void VideoProcessor::blurFrameCUDA(const cv::Mat& src, cv::Mat& dst)
{
    // Before running the kernel for the first time:
    uploadGaussianKernel(0.0f); // or some chosen sigma

    dst.create(src.size(), src.type()); // Ensure dst is allocated, same size/type.

    // We assume 8-bit BGR, so channels = 3. If you handle grayscale or RGBA, adjust accordingly.
    int width    = src.cols;
    int height   = src.rows;
    int channels = src.channels();
    size_t imgSize = width * height * channels * sizeof(unsigned char);

    unsigned char* d_in = nullptr;
    unsigned char* d_out = nullptr;

    // 1. Allocate GPU memory
    cudaMalloc((void**)&d_in,  imgSize);
    cudaMalloc((void**)&d_out, imgSize);

    // 2. Copy input data from CPU (host) to GPU (device)
    cudaMemcpy(d_in, src.data, imgSize, cudaMemcpyHostToDevice);

    // 3. Launch our kernel
    runBlurKernel(d_in, d_out, width, height, channels);

    // 4. Copy the result from GPU back to CPU
    cudaMemcpy(dst.data, d_out, imgSize, cudaMemcpyDeviceToHost);

    // 5. Free GPU memory
    cudaFree(d_in);
    cudaFree(d_out);
}

void VideoProcessor::processCUDA()
{
    if (!isOpened) return;

    cv::Mat frame, output;
    while (true) {
        if (!cap.read(frame) || frame.empty()) {
            break;
        }

        blurFrameCUDA(frame, output);

        writer.write(output);
        // Optionally display
        // cv::imshow("CUDA Processing", output);
        // if (cv::waitKey(1) == 27) { break; }
    }
    cap.release();
    writer.release();
}

