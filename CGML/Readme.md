The projects in this folder compile lower level pieces of the CGML library.

* `Cgml` — C++ DLL which consumes Direct3D 11 API, and exposes a higher level compute-only API.<br/>
It also implements a few CPU-running functions like tensor compressors.
* `ComLightLib` — C++ static library project with the COM [interop library](https://github.com/Const-me/ComLightInterop).<br/>
That library is mostly header only, and used by `Cgml.dll`<br/>
* `SentencePiece` — C++ static library project which contains the corresponding [third-party library](https://github.com/google/sentencepiece).<br/>
That source code fails to compile with C++/20 language version, that's why the separate project.<br/>
That static library is linked into `Cgml.dll`
* `CgmlNet` — .NET DLL which consumes `Cgml.dll`, and exposes a higher level idiomatic C# API.
* `PackShaders` — .NET EXE which implements a custom build step to pack compiled compute shaders, and generate C# helpers to dispatch them.
* `TorchLoader` — .NET DLL which implements loader for ML models in [PyTorch](https://en.wikipedia.org/wiki/PyTorch) format.<br/>
The library only support loading models in that format because `CgmlNet.dll` implements a better C#-targeted serialization format for these models.
 
None of these things is specific to any particular ML model.<br/>
These libraries are designed so that ML models, along with their compute shaders, should be implemented on top of them,
in C# code which consumes these libraries.