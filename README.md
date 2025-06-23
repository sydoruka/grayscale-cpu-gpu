# CUDA Image Grayscale Converter

This project demonstrates how to convert a color image to grayscale using both **CPU** and **GPU (CUDA)** implementations. It compares the performance of each method and outputs the resulting grayscale images.

## Features

- Load image using [stb_image](https://github.com/nothings/stb)
- Grayscale conversion using:
  - CPU implementation (C++)
  - GPU implementation (CUDA)
- Write output images using `stb_image_write`
- Measure and display performance (execution time and speedup)

## Requirements

- CUDA-capable GPU
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- C++ compiler (MSVC, g++, etc.)
- Visual Studio or command-line tools (e.g., `nvcc`)
- `stb_image.h` and `stb_image_write.h` (included)

## How It Works

1. Loads the input image (defined in `IMG_NAME`)
2. Converts the image to grayscale:
   - First on the CPU
   - Then on the GPU using a CUDA kernel
3. Writes both results to disk as PNG files
4. Prints timing and speedup statistics