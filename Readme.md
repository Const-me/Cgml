This project implements a GPU-targeted vendor-agnostic AI library for Windows.<br/>
On top of that library, it implements [Mistral](https://mistral.ai/) model as a demo.

If you find this useful, I’ll be very grateful if you consider a donation to [“Come Back Alive” foundation](https://savelife.in.ua/en/).

# Introduction

From a researcher’s perspective, the modern AI landscape is fascinating.

Modern large language models offer unparalleled capabilities in natural language understanding and generation.
Their vast training datasets enable them to grasp context, generate coherent and contextually relevant text,
and perform a wide range of language-related tasks, from translation to summarization.

Even better, we now have multiple language models with open weights, 
such as [Llama](https://ai.meta.com/llama/), [Mistral](https://mistral.ai/), and others.

However, from a software engineer’s perspective, the AI landscape is a hot mess.

Many critically important AI libraries are developed in close collaboration with NVIDIA,
which uses them as tools to promote sales of CUDA GPUs.
This is notable despite the fact that modern operating systems implement vendor-agnostic APIs to leverage GPGPU hardware,
avoiding reliance on proprietary technologies from GPU vendors.

The community is heavily focused on Python.<br/>
Python is an excellent choice for researchers because of its accessibility, and many university students learn to program in Python.
However, deploying Python in production is hard.<br/>
Furthermore, developing complex software in Python can be challenging.
The dynamically typed nature of the language requires extensive unit testing to achieve optimal software quality.

## CGML Library

In this project, I’ve endeavored to implement a streamlined technology stack for AI.

Here are the main points:

* **Replacement of Python with C#**<br/>
  Python has been replaced with C#. While both languages are high-level, the statically typed nature of C#
  contributes to enhanced software quality.
  Moreover, interfacing with unmanaged C++ code is considerably more straightforward in C# compared to Python.

* **Minimal Runtime Dependencies**<br/>
  The only non-trivial runtime dependency is Microsoft’s .NET runtime.
  As of the latest update, CGML libraries are compiled for .NET 6.0 LTS target.
  The dependency is relatively small; the `windowsdesktop-runtime-6.0.25-win-x64.exe` installer,
  which includes WPF support required by the chat frontend application, is only 55MB.
  This is a fraction of the size of dependencies for torch+cuda.

* **Replacement of CUDA with Direct3D 11**<br/>
  CUDA has been substituted with Direct3D 11.
  While I don't have a Linux PC for testing, implementing another backend based on Vulkan Compute shouldn't be overly challenging.
  Microsoft’s `dxc.exe` compiler, available [here](https://github.com/microsoft/DirectXShaderCompiler),
  can compile HLSL into SPIR-V.
  Portions of DXVK project code, available [here](https://github.com/doitsujin/dxvk), which implements D3D11 on top of Vulkan,
  can be adapted for the new backend. The backend only utilizes a small subset of the D3D11 API,
  including device, immediate context, buffers, compute shaders, and queries.

* **Simplified Approach to Compute Kernels**<br/>
  Conventional AI libraries bury GPU-running compute kernels deep within the lower levels of the stack,
  and require things like Triton to allow developers to program GPUs. CGML takes a different approach.
  It avoids this complexity by requiring users to provide a complete set of compiled compute shaders to implement a specific model.

## Mistral Implementation

The compute shaders are compiled by the `Mistral/MistralShaders` project.<br/>
As you can observe, the entire model requires slightly over 20 shaders.

CPU running components of the model are implemented in the `Mistral/MistralModel` project.<br/>
This involves approximately 3200 lines of C#, with 400 lines automatically generated from HLSL using a custom tool. <br/>
The source code for this tool is located in the `CGML/PackShaders` subfolder of this repository.

While this implementation introduces more code and complexity compared to the Python reference implementation,
I believe it remains within reasonable bounds.

## Desktop Chat Demo

`Mistral/MistralChat` project implements a desktop chat application.<br/>
See another `Readme.md` document in that folder.

The compiled binaries are available on the Releases page in this repository.

## Libraries

The binaries are available on [NuGet](https://www.nuget.org/).<br/>
The package IDs are `Cgml`, `Cgml.TorchLoader`, and `Cgml.MistralModel`.

`Cgml` contains D3D11 backend DLL, and it’s .NET projection.<br/>
You can use it to implement other ML models. Note you going to need the `PackShaders.exe` design-time tool.

`Cgml.TorchLoader` contains PyTorch importer.

`Cgml.MistralModel` contains Mistral model implementation.<br/>
You can use it to integrate the complete model into your software, replacing WPF frontend of `MistralChat.exe`

# Technical Details

## C++ Backend

The C++ DLL implements an API for moving the data to and from the GPU,
along with generic methods for creating and dispatching compute shaders.

All compute shaders are instantiated with a single method call, `iContext.createComputeShaders`.<br/>
Afterwards, they are identified using a 0-based index.

The library assumes that all compute shaders have a single constant buffer bound to the slot #0.<br/>
The `iContext.bindShader` method accepts 0-based index of the shader, and a pointer to the constant buffer data in the system memory.<br/>
Internally, the library maintains a pool of [dynamic](https://learn.microsoft.com/en-us/windows/win32/direct3d11/how-to--use-dynamic-resources)
constant buffers to efficiently marshal these structures to the GPU.

## Shaders Packaging Tool

The custom design-time tool serves the following purposes.

1. **Shader Packaging:** It consolidates all compute shaders for a specific model into a compact binary file.

2. **Runtime Shader Dispatch:** The tool generates the data to dynamically patch the array of shaders at startup,
  based on the optional FP64 support of the end user’s GPU.

3. **Enum Generation:** It generates a strongly-typed C# enum to uniquely identify each compute shader.

4. **C# Language Projection:** The tool parses HLSL source codes and generates the C# representation of the constant buffers required by each shader.
  Additionally, it creates higher-level utility methods for binding shaders, uploading constant buffer data,
  and binding input and output tensors being processed by these compute shaders.

This tool is intended to be utilized as a custom post-build step in the project that compiles these shaders.<br/>
When configured correctly, the Visual Studio IDE should automatically run the tool when necessary.

## BCML1 Codec

The software introduces the BCML1 custom quantized data format for model tensors,
drawing inspiration from [lossy texture codecs](https://learn.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-block-compression)
supported by Direct3D GPUs.

Just like texture block compression, BCML1 is a lossy codec.<br/>
I haven’t deliberately tested the inference accuracy.
Still, based on my tests, the losses introduced by the compressor don’t affect the output of the model.

The codec operates on blocks of 32 numbers, encoding each block into a concise sequence of 20 bytes.<br/>
The structure of the encoded block is as follows:

- **Header (first 4 bytes):** This segment contains two numbers in [FP16 format](https://en.wikipedia.org/wiki/Half-precision_floating-point_format) – a multiplier and an offset.

- **Payload (remaining 16 bytes):** The payload section comprises 32 elements quantized into 4 bits each.
  The decoder utilizes the formula `e = q * multiplier + offset` to compute each tensor element,
  where `q` is an integer in the range [ 0 .. 15 ] extracted from the corresponding 4-bit slice of the block.

For optimal performance, compressed tensors are stored in read-only
[byte address buffers](https://learn.microsoft.com/en-us/windows/win32/direct3d11/direct3d-11-advanced-stages-cs-resources#byte-address-buffer).

Additionally, `uint` elements containing the compressed data are transposed into a sequence of column-major slices of size width*64.<br/>
The first 256 bytes in the byte address buffers are block headers of 64 blocks in the first 64 rows of the original tensor,
the next 256 bytes are first 8 payload elements of these 64 blocks, and so on.<br/>
This approach enhances performance as it ensures that memory loads are fully [coalesced](https://stackoverflow.com/a/5044424/126995).

The compressor, implemented in C++, runs on the CPU during the model import process.
Refer to the `CGML/Cgml/Utils/Compression/bcml1.cpp` source file for details.<br/>
Note that the implementation requires a CPU supporting the AVX2 instruction set.

The decompressor, implemented in HLSL and found in the `Mistral/MistralShaders/bcml.hlsli` include file, operates as a streaming decompressor.
The decoded FP32 numbers remain in registers and never leave the GPU cores.

These two pieces collectively provide an efficient means of encoding and decoding tensors in the BCML1 format.

## Torch Loader

The C# DLL built from the source codes in `CGML/TorchLoader` folder serves as an importer for model weights stored in the PyTorch format.

This library is entirely implemented in C#,
avoiding the use of any code from PyTorch and eliminating the need for the Python runtime to be installed.

## Rotary Position Embedding

To implement [rotary position embedding](https://blog.eleuther.ai/rotary-embeddings/),
the original implementation generates a large tensor on startup.<br/>
The elements of that tensor are complex numbers in FP64 precision, which immediately kills compatibility with some current-generation GPUs.
Specifically, [Intel Arc](https://en.wikipedia.org/wiki/Intel_Arc) GPUs don’t support FP64 arithmetic instructions.

In my implementation the magic numbers are computed on demand 
with [`sincos`](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-sincos)
intrinsic, and FP64 hardware support is optional for that shader.

See `rotaryEmbedding.hlsl` compute shader for the implementation.

There's a temporary utility in `Mistral/Temp/Freqs` subfolder which validates the formula I’m using in that shader,
by comparing numbers with the `*.tsv` table I saved from the reference implementation written in Python.

## Random Sampling Shaders

Sampling shaders, `sampleTopP.hlsl` for the original Mistral model and `sampleTopK.hlsl` for the Instruct 0.2 model,
are using an interesting trick to sort a long vector of floats in [O(N)](https://en.wikipedia.org/wiki/Big_O_notation) time.
The trick is called “[counting sort](https://en.wikipedia.org/wiki/Counting_sort)”.
That Wikipedia article says that’s an integer sorting algorithm only suitable for small integer keys.
However, with FP16 precision everywhere in Mistral implementation, float numbers can be viewed as small integers.

Because the input probabilities are non-negative, the input vector only contains up to 0x8000 = 32768 unique floats.<br/>
To implement the sampling, these shaders are using a temporary buffer with 0x8000 `uint` elements.
That buffer only takes 128kb of VRAM, very reasonable amount of data which should fit in L2 cache of most GPUs.

The sorting algorithm has the following steps,
with [`AllMemoryBarrierWithGroupSync()`](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/allmemorybarrierwithgroupsync)
between the steps:

1. Fill temporary buffer with zeros.

2. Load input vector, count elements with [`InterlockedAdd()`](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/interlockedadd)
  intrinsics, incrementing elements of the temporary buffer in global memory.

3. Compute **exclusive** [prefix sum](https://en.wikipedia.org/wiki/Prefix_sum) of the temporary buffer, in-place.<br/>
  Because [wave intrinsics](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/waveprefixsum) aren’t available in D3D11,
  the implementation uses group shared memory for this step.

4. Load input vector once again, and find sorted position of elements
  using another variant of `InterlockedAdd()` which outputs original input values.

On my desktop computer with nVidia 1080Ti, `sampleTopK.hlsl` compute shader takes less than 250 microseconds per dispatch.

# Licensing

DLL projects are covered by LGPL 2.1 license.

MistralChat desktop app is GPLv2.

See also `Pre-existing IP.md` document in this repository.