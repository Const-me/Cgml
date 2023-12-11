#pragma once
#include "TensorShape.h"

namespace Cgml
{
	// Element type for all these tensors
	enum struct eDataType : uint8_t
	{
		// Half-precision floats, https://en.wikipedia.org/wiki/Half-precision_floating-point_format
		FP16 = 0,
		// 32-bit floats
		FP32 = 1,
		// 32-bit integers
		U32 = 2,
		// Non-standard half-precision floats, https://en.wikipedia.org/wiki/Bfloat16_floating-point_format
		BF16 = 3,
	};

	// Loosely corresponds to D3D11_USAGE enumeration
	enum struct eBufferUse : uint8_t
	{
		// Immutable tensor, readable from GPU
		Immutable = 0,
		// Read+write tensor, readable and writable on GPU
		ReadWrite = 1,
		// Read+write tensor, readable and writable on GPU, which supports downloads from GPU
		ReadWriteDownload = 2,
		// The tensor is accessible by both GPU (read only) and CPU (write only). Optimized for resources frequently updated from CPU.
		Dynamic = 3,
	};

	// VRAM layout of the tensor
	enum struct eTensorLayout : uint8_t
	{
		// The tensor is dense, and uncompressed
		Dense = 0,
		// BCML1 quantized: 32 tensor elements => 20 bytes of data
		BCML1 = 1,
	};

	struct sTensorDesc
	{
		// Size and stride
		TensorShape shape;

		/// <summary>Type of elements</summary>
		eDataType dataType;

		/// <summary>Usage flags for the GPU buffer</summary>
		eBufferUse usage;

		/// <summary>VRAM layout of the tensor</summary>
		eTensorLayout layout;
	};

	using pfnUpdateDynamicTensor = HRESULT( __stdcall* )( void* rdi, uint32_t cb, void* pv );

	using pfnReadTensor = HRESULT( __stdcall* )( const void* rsi, uint32_t cb, void* pv );

	enum struct eLoadTransform: uint8_t
	{
		// Don't transform anything
		None = 0,
		// Convert BF16 numbers into IEEE FP16
		Fp16MakeIeee = 1,
	};
}