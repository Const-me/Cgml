GPU-targeted vendor-agnostic AI library for Windows.

For unmanaged interop, it depends on [ComLightInterop](https://github.com/Const-me/ComLightInterop) library.

This library doesn’t include any compute shaders, and is not specific to any ML model.<br/>
ML models are expected to be implemented at the higher level of the stack, in a project which consumes this DLL.

Instead, this project only contains low-level utilities to initialize Direct3D 11,
create a set of compute shaders implementing a model, move tensors data between system memory and VRAM, 
and dispatch compute shaders passing tensors to read and write, and a single constant buffer.

It also implements serializer which keeps multiple tensors in a single ZIP archive, and a few more utility functions and classes.

Because the underlying Cgml.dll C++ DLL is only built for Win64 platform, this library will only run when used from a 64-bit process.