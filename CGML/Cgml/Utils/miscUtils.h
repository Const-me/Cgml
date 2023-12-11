#pragma once
#include <intrin.h>

inline __m128i __vectorcall load( const std::array<uint32_t, 4>& arr )
{
	return _mm_loadu_si128( ( const __m128i* )arr.data() );
}

inline bool __vectorcall vectorEqual( __m128i a, __m128i b )
{
	__m128i xx = _mm_xor_si128( a, b );
	return (bool)_mm_testz_si128( xx, xx );
}

// Interpret the vector as uint32_t lanes, compute product of all lanes
inline size_t horizontalProduct( __m128i vec )
{
	__m128i shifted = _mm_srli_si128( vec, 4 );
	__m128i products = _mm_mul_epu32( vec, shifted );

	uint64_t high = (uint64_t)_mm_extract_epi64( products, 1 );
	uint64_t low = (uint64_t)_mm_cvtsi128_si64( products );
	return high * low;
}

inline __m128i __vectorcall setLow_size( size_t low )
{
	return _mm_cvtsi64_si128( (int64_t)low );
}
inline __m128i __vectorcall setr_size( size_t low, size_t high )
{
	__m128i v = setLow_size( low );
	v = _mm_insert_epi64( v, (int64_t)high, 1 );
	return v;
}
inline __m128i __vectorcall setHigh_size( size_t high )
{
	__m128i v = _mm_setzero_si128();
	v = _mm_insert_epi64( v, (int64_t)high, 1 );
	return v;
}

__m128i __vectorcall bufferMemoryUsage( ID3D11Buffer* buffer );
__m128i __vectorcall resourceMemoryUsage( ID3D11ShaderResourceView* srv );

namespace Cgml
{
	struct TensorShape;
}

// True when we can multiply two tensors of the provided shapes
bool canMulMat( const Cgml::TensorShape& t0, const Cgml::TensorShape& t1 );

// Make a mask for _mm256_maskload_ps or _mm256_maskstore_ps instruction to handle remainder elements
inline __m256i makeAvxMask( size_t rem )
{
	assert( rem > 0 && rem < 8 );
	uint64_t scalar = ~(uint64_t)0;
	rem = ( 8 - rem ) * 8;
	scalar = scalar >> rem;
	__m128i v = _mm_cvtsi64_si128( (int64_t)scalar );
	return _mm256_cvtepi8_epi32( v );
}

inline HRESULT getLastHr()
{
	return HRESULT_FROM_WIN32( ::GetLastError() );
}

// Scale time in seconds from unsigned 64 bit rational number ( mul / div ) into 100-nanosecond ticks
// These 100-nanosecond ticks are used in NTFS, FILETIME, .NET standard library, media foundation, and quite a few other places
inline uint64_t makeTime( uint64_t mul, uint64_t div )
{
	mul *= 10'000'000;
	mul += ( ( div / 2 ) - 1 );
	return mul / div;
}

inline bool checkAvx2Support()
{
	// https://en.wikipedia.org/wiki/CPUID#EAX=7,_ECX=0:_Extended_Features
	int cpuInfo[ 4 ];
	__cpuid( cpuInfo, 7 );
	return ( cpuInfo[ 1 ] & ( 1 << 5 ) ) != 0;
}

inline bool checkBmi2Suppport()
{
	// https://en.wikipedia.org/wiki/CPUID#EAX=7,_ECX=0:_Extended_Features
	int cpuInfo[ 4 ];
	__cpuid( cpuInfo, 7 );
	return ( cpuInfo[ 1 ] & ( 1 << 8 ) ) != 0;
}

inline bool checkF16cSuppport()
{
	// https://en.wikipedia.org/wiki/CPUID#EAX=1:_Processor_Info_and_Feature_Bits
	int cpuInfo[ 4 ];
	__cpuid( cpuInfo, 1 );
	return ( cpuInfo[ 2 ] & ( 1 << 29 ) ) != 0;
}