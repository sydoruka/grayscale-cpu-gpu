#include <iostream>
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

#include "stb_image.h"
#include "stb_image_write.h"

#define IMG_NAME "YOUR_IMAGE_PATH_HERE"

struct Pixel
{
    unsigned char r, g, b, a;
};

void ConvertImageToGrayCpu(unsigned char* imageRGBA, int width, int height)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            Pixel* ptrPixel = (Pixel*)&imageRGBA[y * width * 4 + x * 4];

            unsigned char pixelValue = static_cast<unsigned char>((ptrPixel->r + ptrPixel->g + ptrPixel->b) / 3);

            ptrPixel->r = pixelValue;
            ptrPixel->g = pixelValue;
            ptrPixel->b = pixelValue;
            ptrPixel->a = 255;
        }
    }
}

__global__ void ConvertImageToGrayGpu(unsigned char* imageRGBA, int width, int height)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        unsigned int idx = y * width + x;
        Pixel* ptrPixel = (Pixel*)&imageRGBA[idx * 4];

        unsigned char pixelValue = static_cast<unsigned char>((ptrPixel->r + ptrPixel->g + ptrPixel->b) / 3);

        ptrPixel->r = pixelValue;
        ptrPixel->g = pixelValue;
        ptrPixel->b = pixelValue;
        ptrPixel->a = 255;
    }
}

int main() {
    // Loading image file
    int width, height, componentCount;
    std::cout << "Loading image file...";
    unsigned char* imageDataCPU = stbi_load(IMG_NAME, &width, &height, &componentCount, 4);
    unsigned char* imageDataGPU = stbi_load(IMG_NAME, &width, &height, &componentCount, 4);

    // Check if stbi_load failed
    if (!imageDataCPU && !imageDataGPU) {
        std::cerr << "Failed to open " << IMG_NAME << std::endl;
        return -1;
    }
    std::cout << " DONE" << std::endl;

    // Start CPU time measurement
    auto startTimeCPU = std::chrono::high_resolution_clock::now();

    // Process image on CPU
    std::cout << "Processing image on CPU...";
    ConvertImageToGrayCpu(imageDataCPU, width, height);
    std::cout << " DONE" << std::endl;

    // End CPU time measurement
    auto endTimeCPU = std::chrono::high_resolution_clock::now();
    auto durationCPU = std::chrono::duration_cast<std::chrono::nanoseconds>(endTimeCPU - startTimeCPU);

    // Build CPU output filename
    std::string fileNameOut = IMG_NAME;
    fileNameOut = fileNameOut.substr(0, fileNameOut.find_last_of('.')) + "_CPU_gray.jpg";

    // Allocate Memory for GPU
    unsigned char* ptrImageDataGpu = nullptr;
    cudaError_t cudaStatus = cudaMalloc(&ptrImageDataGpu, width * height * 4);

    if (cudaStatus != cudaSuccess) {
        std::cerr << "Error: cudaMalloc failed with code " << cudaStatus << ": " << cudaGetErrorString(cudaStatus) << std::endl;
        return -1;
    }

    // Copy data to GPU
    std::cout << "Copying initial image data to GPU...";
    cudaMemcpy(ptrImageDataGpu, imageDataGPU, width * height * 4, cudaMemcpyHostToDevice);
    std::cout << " DONE" << std::endl;

    // Start GPU time measurement
    auto startTimeGPU = std::chrono::high_resolution_clock::now();

    // Process image on GPU
    std::cout << "Running CUDA Kernel...";
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    ConvertImageToGrayGpu <<<gridSize, blockSize >>> (ptrImageDataGpu, width, height);

    cudaDeviceSynchronize();  // Ensure the kernel completes before timing

    cudaError_t syncError = cudaGetLastError();
    if (syncError != cudaSuccess) {
        std::cerr << "CUDA error after kernel execution: " << cudaGetErrorString(syncError) << std::endl;
        return -1;
    }

    cudaError_t asyncError = cudaDeviceSynchronize();  // Check for asynchronous errors
    if (asyncError != cudaSuccess) {
        std::cerr << "CUDA asynchronous error after synchronization: " << cudaGetErrorString(asyncError) << std::endl;
        return -1;
    }
    std::cout << " DONE" << std::endl;

    // End GPU time measurement
    auto endTimeGPU = std::chrono::high_resolution_clock::now();
    auto durationGPU = std::chrono::duration_cast<std::chrono::nanoseconds>(endTimeGPU - startTimeGPU);

    // Copy data from GPU
    std::cout << "Copying data from GPU...";
    cudaMemcpy(imageDataGPU, ptrImageDataGpu, width * height * 4, cudaMemcpyDeviceToHost);
    std::cout << " DONE" << std::endl;


    // Build GPU output filename
    std::string fileNameOutGPU = IMG_NAME;
    fileNameOutGPU = fileNameOutGPU.substr(0, fileNameOutGPU.find_last_of('.')) + "_GPU_gray.jpg";

    // Write CPU image back to disk
    std::cout << "Writing CPU image to disk...";
    stbi_write_png(fileNameOut.c_str(), width, height, 4, imageDataCPU, 4 * width);
    std::cout << " DONE" << std::endl;

    // Write GPU image back to disk
    std::cout << "Writing GPU image to disk...";
    stbi_write_png(fileNameOutGPU.c_str(), width, height, 4, imageDataGPU, 4 * width);
    std::cout << " DONE\n" << std::endl;

    // Display performances
    std::cout << "Time taken to process on CPU: " << durationCPU.count() << " nanoseconds; "
        << durationCPU.count() / 1e9 << " seconds" << std::endl;
    std::cout << "Time taken to process on GPU: " << durationGPU.count() << " nanoseconds; "
        << durationGPU.count() / 1e9 << " seconds" << std::endl;

    // Calculate and display speedup factor
    double speedup = static_cast<double>(durationCPU.count()) / durationGPU.count();
    std::cout << "Speedup factor (CPU to GPU): " << round(speedup) << " times faster" << std::endl;

    // Free Memory
    cudaFree(ptrImageDataGpu);
    stbi_image_free(imageDataCPU);
    stbi_image_free(imageDataGPU);

    return 0;
}
