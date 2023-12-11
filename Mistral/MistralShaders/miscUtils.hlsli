// Try to emulate the arithmetics Pytorch is doing with CUDA
// Setting this to 0 should slightly improve both precision and performance
#define CUDA_COMPAT 1

inline uint hadd( uint2 v )
{
	return v.x + v.y;
}

inline uint hadd( uint3 v )
{
	return v.x + v.y + v.z;
}

#ifndef USE_BF16
#define USE_BF16 1
#endif

#if USE_BF16

#define Tensor Buffer<uint>
#define OutputTensor RWBuffer<uint>
#define StoredType uint

inline float load( Buffer<uint> t, uint idx )
{
	return asfloat( t[ idx ] << 16 );
}
inline float load( RWBuffer<uint> t, uint idx )
{
	return asfloat( t[ idx ] << 16 );
}

inline uint roundBf16Nearest( float f )
{
	// Scalar version:
	// uint32_t rounding_bias = ((U32 >> 16) & 1) + UINT32_C(0x7FFF);
	// output_row[ col ] = static_cast<uint16_t>( ( U32 + rounding_bias ) >> 16 );
	const uint u = asuint( f );
	const uint bias = ( u & 0x10000u ) ? 0x8000 : 0x7FFF;
	return ( u + bias ) >> 16;
}

inline void store( RWBuffer<uint> tensor, uint idx, float f )
{
	tensor[ idx ] = roundBf16Nearest( f );
}

#else

#define Tensor Buffer<float>
#define OutputTensor RWBuffer<float>
#define StoredType float

inline float load( Buffer<float> t, uint idx )
{
	return t[ idx ];
}
inline float load( RWBuffer<float> t, uint idx )
{
	return t[ idx ];
}

// This function rounds FP32 value to the nearest FP16, using bankers rounding
// When GPUs are converting FP32 to FP16, they always truncate towards 0, documented there:
// https://learn.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-data-conversion#conververting-from-a-higher-range-representation-to-a-lower-range-representation
inline float roundFp16Nearest( const float src )
{
	const uint trunc16 = f32tof16( src );
	const float trunc32 = f16tof32( trunc16 );

	const uint truncExp = ( trunc16 >> 10 ) & 0x1F;
	if( truncExp != 0x1F )
	{
		const uint next16 = trunc16 + 1;
		const float next32 = f16tof32( next16 );

		const float errTrunc = abs( src - trunc32 );
		const float errNext = abs( src - next32 );

		if( errTrunc < errNext )
		{
			// Truncated was closer to the source
			return trunc32;
		}
		else if( errTrunc > errNext )
		{
			// Truncated + 1 was closer to the source
			return next32;
		}
		else
		{
			// Exactly half, doing banker's rounding to nearest even
			return ( 0 == ( trunc16 & 1 ) ) ? trunc32 : next32;
		}
	}
	else
	{
		// INF or NAN
		return src;
	}
}

inline void store( RWBuffer<float> tensor, uint idx, float f )
{
	tensor[ idx ] = roundFp16Nearest( f );
}
#endif