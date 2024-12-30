#include "VideoProcessor.hpp"
#include <iostream>
#include <chrono>

int main(int argc, char* argv[])
{
    // In real code, parse arguments or read from config
    std::string inputPath  = "test_videos/input.mp4";
    std::string outputPath1 = "test_videos/output_single_thread.mp4";
    std::string outputPath2 = "test_videos/output_multi_thread.mp4";
    std::string outputPathCUDA = "test_videos/output_cuda.mp4";

    // Single-threaded processing
    {
        VideoProcessor vp(inputPath, outputPath1);

        auto start = std::chrono::high_resolution_clock::now();
        vp.processSingleThreaded();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Single-threaded processing took " << elapsed.count() << " seconds.\n";
    }

    //Multi-threaded processing
    {
        unsigned int numThreads = 4;
        VideoProcessor vp(inputPath, outputPath2);

        auto start = std::chrono::high_resolution_clock::now();
        vp.processMultiThreaded(numThreads);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Multi-threaded processing with " << numThreads
                  << " threads took " << elapsed.count() << " seconds.\n";
    }

    //CUDA processing
    {
        VideoProcessor vp(inputPath, outputPathCUDA);

            auto start = std::chrono::high_resolution_clock::now();
            vp.processCUDA();
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            std::cout << "CUDA processing took " << elapsed.count() << " seconds.\n";
    }

    return 0;
}
