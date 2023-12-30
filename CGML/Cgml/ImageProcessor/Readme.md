The source codes in this folder implement a better equivalent of 
Python’s [`BlipImageProcessor`](https://huggingface.co/docs/transformers/main/en/model_doc/blip#transformers.BlipImageProcessor) class.

Here's the differences.

* Image decoder uses [WIC](https://learn.microsoft.com/en-us/windows/win32/wic/-wic-about-windows-imaging-codec), image codecs are shipped with Windows.

* This image processor is implemented as a set of shaders and runs on GPU.<br/>
  This way is much faster than [Pillow](https://pypi.org/project/Pillow/) library used by the Python version.

* This version generates full set of mip levels for the input image, then uses high-quality anisotropic sampling to resize the image.<br/>
  The approach is _the_ best way to resize raster images to smaller sizes.<br/>
  For optimal accuracy, the mipmapped intermediate texture even uses FP16 precision for the pixels.

The shaders used in the processor are compiled with `ImageShaders` project.<br/>
Then, `ImageShadersPack` custom tool compresses these binaries with LZ4, and generates C++ inline file with `std::array` of bytes.<br/>
Implemented this way to save development time.<br/>
For boring technical reasons (no wave intrinsics until D3D12, conflicting `D3D11_BIND_RENDER_TARGET` and `D3D11_BIND_UNORDERED_ACCESS` binding fllags),
the implementation is mostly based on pixel shaders, as opposed to compute.<br/>
Designing COM API to expose graphics-related parts of D3D11 API to C# would take a lot of time for very questionable benefits.