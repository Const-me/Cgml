This project compiles a few shaders used by the `iContext.loadImage` implementation.

For optimal quality, that method requires [anisotropic sampling](https://en.wikipedia.org/wiki/Anisotropic_filtering).

The compute shaders have [Texture2D.SampleGrad](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-to-samplegrad) available,
but it's very hard to use without [wave intrinsics](https://github.com/Microsoft/DirectXShaderCompiler/wiki/Wave-Intrinsics).<br/>
CGML is based on Direct3D 11, which means wave intrinsics are unavailable.

To workaround, `iContext.loadImage` is implemented with graphics pipeline, by rendering full-screenn triangles.

I didn’t want to expose graphics pipeline to C#, this would be too much work for too little profit.

That’s why the graphics pipeline is contained to the C++ side of the interop, and `Cgml.dll` includes a few shaders which implement the feature.