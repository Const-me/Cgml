GPU-targeted vendor-agnostic AI library for Windows.

For unmanaged interop, it depends on [ComLightInterop](https://github.com/Const-me/ComLightInterop) library.

This library doesn’t include any compute shaders, and is not specific to any ML model.<br/>
ML models are expected to be implemented at the higher level of the stack, in a project which consumes this DLL.

Instead, this project only contains low-level utilities to initialize Direct3D 11,
create a set of compute shaders implementing a model, move tensors data between system memory and VRAM, 
and dispatch compute shaders passing tensors to read and write, and a single constant buffer.

It also implements serializer which keeps multiple tensors in a single ZIP archive, and a few more utility functions and classes.

Because the underlying Cgml.dll C++ DLL is only built for Win64 platform, this library will only run when used from a 64-bit process.

# Tensor Conventions

This library uses programmer-friendly approach to these tensors.<br/>
This differs from Python libraries like PyTorch and NumPy which were designed for mathematical conventions
but then in practice they almost exclusively using negative numbers to index tensor dimensions, to count them from the right.

By default, all tensors are row major.<br/>
A matrix with 3 columns and 2 rows is represented as a tensor of size `[ 3, 2, 1, 1 ]`

The length of the shape is irrelevant in CGML.<br/>
A matrix with 7 columns and 1 row is indistinguishable from a vector of length 7.<br/>
This behaviour is by design.

There’s a limit for the number of dimensions in a tensor, that limit is 4.<br/>
This allows to keep tensor sizes in 128-bit SIMD vectors.<br/>
This also eliminates dynamic memory allocations and pointer chasing while manipulating shapes of the tensors.

For hardware compatibility reasons, there’s no support for FP64 floats or int64 integers in the tensors.<br/>
Luckily, for ML applications FP32 floats and 32-bit integers are sufficient.

## FP16 Conventions

The library has some support for both flavours of FP16, IEEE 754 and BF16.<br/>
However, they both need special handling in the HLSL shaders on your side.

### Half-precision floating-point

IEEE 754 FP16 tensors are exposed to the shaders as `Buffer<float>` objects for inputs, or `RWBuffer<float>` objects for outputs.

The problem with that, `RWBuffer<float>` unordered access views are always rounding towards zero when storing values into FP16 buffers.
This is [documented](https://learn.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-data-conversion#conververting-from-a-higher-range-representation-to-a-lower-range-representation) by Microsoft,
but that’s probably not what you want.

Instead, your shaders should round to nearest FP16 when storing values into the output FP16 tensors.<br/>
Here’s a function for that, which I carefully unit-tested against `vcvtps2ph` CPU instruction on full range of floats, excluding NAN values.

```
// This function rounds FP32 value to the nearest FP16, using bankers rounding
// When GPUs are converting FP32 to FP16, they always truncate towards 0, documented there:
// https://learn.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-data-conversion#conververting-from-a-higher-range-representation-to-a-lower-range-representation
inline float roundFp16Nearest( float src )
{
	[branch]
	if( abs( src ) < 65520.0f )
	{
		const uint truncatedFp16 = f32tof16( src );
		const float truncated = f16tof32( truncatedFp16 );
		const float next = f16tof32( truncatedFp16 + 1 );

		const float errTrunc = abs( src - truncated );
		const float errNext = abs( src - next );

		if( errTrunc < errNext )
		{
			// Truncated was closer to the source
			return truncated;
		}
		else if( errTrunc > errNext )
		{
			// Truncated + 1 was closer to the source
			return next;
		}
		else
		{
			// Exactly half, doing banker's rounding to nearest even
			return ( 0 == ( truncatedFp16 & 1 ) ) ? truncated : next;
		}
	}
	else
	{
		// Return +inf or -inf depending on the sign bit of the input
		// Note this destroys NAN values, converting them to inf as well
		uint u = asuint( src );
		u &= 0x80000000u;
		u |= 0x7f800000u;
		return asfloat( u );
	}
}
```

### bfloat16 floating point

BF16 tensors are exposed to the shaders as `Buffer<uint>` objects for inputs, or `RWBuffer<uint>` objects for outputs.<br/>
You should convert these elements to/from floats yourself.

Converting BF16 to float only takes a single bitwise shift instruction: `asfloat( bf << 16 )`

Downcasting is harder due to rounding. Here’s one possible HLSL implementation.

```
inline uint roundBf16Nearest( float f )
{
	// Scalar version:
	// uint32_t rounding_bias = ((U32 >> 16) & 1) + UINT32_C(0x7FFF);
	// output_row[ col ] = static_cast<uint16_t>( ( U32 + rounding_bias ) >> 16 );
	const uint u = asuint( f );
	const uint bias = ( u & 0x10000u ) ? 0x8000 : 0x7FFF;
	return ( u + bias ) >> 16;
}
```