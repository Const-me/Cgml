// The source code in this file implements dbgTensorsDiff() DLL entry point, and requires AVX2 and F16C ISA extensions
// It is only useful for debugging and QA, it should not be called in production
#include <stdafx.h>
#include "../API/sTensorDesc.h"
#include "../../ComLightLib/hresult.h"
#include "../D3D/tensorUtils.h"
#include <ammintrin.h>

namespace Cgml
{
	struct TensorsDiff
	{
		float maxAbsDiff;
		float avgAbsDiff;
		float rms;
	};

	struct sTensorBuffer
	{
		sTensorDesc desc;
		uint32_t lengthBytes;
	};
}

using namespace Cgml;

namespace
{
	inline float horizontalMax( __m256 v8 )
	{
		__m128 v = _mm256_extractf128_ps( v8, 1 );
		v = _mm_max_ps( v, _mm256_castps256_ps128( v8 ) );
		v = _mm_max_ps( v, _mm_movehl_ps( v, v ) );
		v = _mm_max_ss( v, _mm_movehdup_ps( v ) );
		return _mm_cvtss_f32( v );
	}

	inline double horizontalSum( __m256 v8 )
	{
		__m128 high = _mm256_extractf128_ps( v8, 1 );
		__m128 low = _mm256_castps256_ps128( v8 );
		__m256d high64 = _mm256_cvtps_pd( high );
		__m256d low64 = _mm256_cvtps_pd( low );

		__m256d v4 = _mm256_add_pd( high64, low64 );
		__m128d v2 = _mm256_extractf128_pd( v4, 1 );
		v2 = _mm_add_pd( v2, _mm256_castpd256_pd128( v4 ) );
		v2 = _mm_add_sd( v2, _mm_unpackhi_pd( v2, v2 ) );
		return _mm_cvtsd_f64( v2 );
	}

	class DiffImpl
	{
		__m256 maxAbsDiff = _mm256_setzero_ps();
		__m256 totalAbsDiff = _mm256_setzero_ps();
		__m256 sumSquares = _mm256_setzero_ps();
		size_t countElements = 0;

	public:

		__forceinline void add( __m256 a, __m256 b, size_t elts = 8 )
		{
			__m256 diff = _mm256_sub_ps( a, b );
			sumSquares = _mm256_fmadd_ps( diff, diff, sumSquares );
			const __m256 signBit = _mm256_set1_ps( -0.0f );
			diff = _mm256_andnot_ps( signBit, diff );
			maxAbsDiff = _mm256_max_ps( maxAbsDiff, diff );
			totalAbsDiff = _mm256_add_ps( totalAbsDiff, diff );
			countElements += elts;
		}

		HRESULT reduce( TensorsDiff& rdi ) const
		{
			rdi.maxAbsDiff = horizontalMax( maxAbsDiff );
			__m128d sum = _mm_setr_pd( horizontalSum( sumSquares ), horizontalSum( totalAbsDiff ) );
			sum = _mm_div_pd( sum, _mm_set1_pd( (double)(int64_t)countElements ) );
			sum = _mm_sqrt_sd( sum, sum );
			const __m128 sum32 = _mm_cvtpd_ps( sum );
			rdi.avgAbsDiff = _mm_cvtss_f32( _mm_movehdup_ps( sum32 ) );
			rdi.rms = _mm_cvtss_f32( sum32 );
			return S_OK;
		}
	};

	// Load 8 FP16 numbers, and upcast to FP32
	__forceinline __m256 loadfp16( const uint16_t* rsi )
	{
		__m128i v = _mm_loadu_si128( ( const __m128i* )rsi );
		return _mm256_cvtph_ps( v );
	}

	// Load 8 BF16 numbers, and upcast to FP32
	__forceinline __m256 loadbf16( const uint16_t* rsi )
	{
		__m128i v = _mm_loadu_si128( ( const __m128i* )rsi );
		__m256i res = _mm256_cvtepu16_epi32( v );
		res = _mm256_slli_epi32( res, 16 );
		return _mm256_castsi256_ps( res );
	}

	HRESULT fp16( TensorsDiff& rdi, const uint16_t* a, const uint16_t* b, size_t length )
	{
		const uint16_t* const aEndAligned = a + _andn_u64( 7, length );
		const size_t rem = length & 7;
		DiffImpl diff;
		while( a < aEndAligned )
		{
			__m256 av = loadfp16( a );
			a += 8;
			__m256 bv = loadfp16( b );
			b += 8;
			diff.add( av, bv );
		}

		if( 0 != rem )
		{
			uint16_t buffer[ 16 ];
			_mm256_storeu_si256( ( __m256i* )buffer, _mm256_setzero_si256() );
			__movsw( buffer, a, rem );
			__movsw( buffer + 8, b, rem );

			__m256 av = loadfp16( buffer );
			__m256 bv = loadfp16( buffer + 8 );
			diff.add( av, bv, rem );
		}

		return diff.reduce( rdi );
	}

	HRESULT bf16( TensorsDiff& rdi, const uint16_t* a, const uint16_t* b, size_t length )
	{
		const uint16_t* const aEndAligned = a + _andn_u64( 7, length );
		const size_t rem = length & 7;
		DiffImpl diff;
		while( a < aEndAligned )
		{
			__m256 av = loadbf16( a );
			a += 8;
			__m256 bv = loadbf16( b );
			b += 8;
			diff.add( av, bv );
		}

		if( 0 != rem )
		{
			uint16_t buffer[ 16 ];
			_mm256_storeu_si256( ( __m256i* )buffer, _mm256_setzero_si256() );
			__movsw( buffer, a, rem );
			__movsw( buffer + 8, b, rem );

			__m256 av = loadbf16( buffer );
			__m256 bv = loadbf16( buffer + 8 );
			diff.add( av, bv, rem );
		}

		return diff.reduce( rdi );
	}

	HRESULT fp32( TensorsDiff& rdi, const float* a, const float* b, size_t length )
	{
		const float* const aEndAligned = a + _andn_u64( 7, length );
		const size_t rem = length & 7;
		DiffImpl diff;
		while( a < aEndAligned )
		{
			__m256 av = _mm256_loadu_ps( a );
			a += 8;
			__m256 bv = _mm256_loadu_ps( b );
			b += 8;
			diff.add( av, bv );
		}

		if( 0 != rem )
		{
			const __m256i loadMask = makeAvxMask( rem );
			__m256 av = _mm256_maskload_ps( a, loadMask );
			__m256 bv = _mm256_maskload_ps( b, loadMask );
			diff.add( av, bv, rem );
		}

		return diff.reduce( rdi );
	}

	HRESULT fpMixed( TensorsDiff& rdi, const float* a, const uint16_t* b, size_t length )
	{
		const float* const aEndAligned = a + _andn_u64( 7, length );
		const size_t rem = length & 7;
		DiffImpl diff;
		while( a < aEndAligned )
		{
			__m256 av = _mm256_loadu_ps( a );
			a += 8;
			__m256 bv = loadfp16( b );
			b += 8;
			diff.add( av, bv );
		}

		if( 0 != rem )
		{
			const __m256i loadMask = makeAvxMask( rem );
			__m256 av = _mm256_maskload_ps( a, loadMask );

			uint16_t buffer[ 8 ];
			_mm_storeu_si128( ( __m128i* )buffer, _mm_setzero_si128() );
			__movsw( buffer, b, rem );

			__m256 bv = loadfp16( buffer );

			diff.add( av, bv, rem );
		}

		return diff.reduce( rdi );
	}

	HRESULT fp16bf16mixed( TensorsDiff& rdi, const uint16_t* a, const uint16_t* b, size_t length )
	{
		const uint16_t* const aEndAligned = a + _andn_u64( 7, length );
		const size_t rem = length & 7;
		DiffImpl diff;
		while( a < aEndAligned )
		{
			__m256 av = loadfp16( a );
			a += 8;
			__m256 bv = loadbf16( b );
			b += 8;
			diff.add( av, bv );
		}

		if( 0 != rem )
		{
			uint16_t buffer[ 16 ];
			_mm256_storeu_si256( ( __m256i* )buffer, _mm256_setzero_si256() );
			__movsw( buffer, a, rem );
			__movsw( buffer + 8, b, rem );

			__m256 av = loadfp16( buffer );
			__m256 bv = loadbf16( buffer + 8 );
			diff.add( av, bv, rem );
		}

		return diff.reduce( rdi );
	}

	class TensorData
	{
		const uint8_t* ptr = nullptr;
		size_t sourceBufferBytes = 0;
		uint8_t cbElement = 0;
		sTensorDesc desc;
	public:

		HRESULT initialize( const uint8_t* p, const sTensorBuffer& d )
		{
			ptr = p;
			sourceBufferBytes = d.lengthBytes;
			desc = d.desc;

			sTypeInfo ti;
			CHECK( sTypeInfo::initialize( ti, desc ) );
			cbElement = ti.cbElement;

			if( desc.shape.countElements() * (size_t)cbElement != sourceBufferBytes )
			{
				logError( u8"dbgTensorsDiff does not support padded tensors" );
				return E_NOTIMPL;
			}

			return S_OK;
		}

		static HRESULT diff( TensorsDiff& rdi, const TensorData& a, const TensorData& b )
		{
			if( !TensorShape::isSameShape( a.desc.shape, b.desc.shape ) )
			{
				logError( u8"The tensors have different size" );
				return E_INVALIDARG;
			}

			if( a.desc.dataType == eDataType::FP16 && b.desc.dataType == eDataType::FP16 )
				return fp16( rdi, (const uint16_t*)a.ptr, (const uint16_t*)b.ptr, a.sourceBufferBytes / 2 );

			if( a.desc.dataType == eDataType::FP32 && b.desc.dataType == eDataType::FP32 )
				return fp32( rdi, (const float*)a.ptr, (const float*)b.ptr, a.sourceBufferBytes / 4 );

			if( a.desc.dataType == eDataType::BF16 && b.desc.dataType == eDataType::BF16 )
				return bf16( rdi, (const uint16_t*)a.ptr, (const uint16_t*)b.ptr, a.sourceBufferBytes / 2 );

			if( a.desc.dataType == eDataType::FP32 && b.desc.dataType == eDataType::FP16 )
				return fpMixed( rdi, (const float*)a.ptr, (const uint16_t*)b.ptr, a.sourceBufferBytes / 4 );

			if( a.desc.dataType == eDataType::FP16 && b.desc.dataType == eDataType::FP32 )
				return fpMixed( rdi, (const float*)b.ptr, (const uint16_t*)a.ptr, a.sourceBufferBytes / 2 );

			if( a.desc.dataType == eDataType::BF16 && b.desc.dataType == eDataType::FP16 )
				return fp16bf16mixed(rdi, (const uint16_t*)b.ptr, (const uint16_t*)a.ptr, a.sourceBufferBytes / 2 );

			if( a.desc.dataType == eDataType::FP16 && b.desc.dataType == eDataType::BF16 )
				return fp16bf16mixed( rdi, (const uint16_t*)a.ptr, (const uint16_t*)b.ptr, a.sourceBufferBytes / 2 );

			return E_NOTIMPL;
		}
	};
}

HRESULT dbgTensorsDiff( TensorsDiff& rdi, const uint8_t* a, const sTensorBuffer& aDesc, const uint8_t* b, const sTensorBuffer& bDesc )
{
	if( aDesc.desc.layout != eTensorLayout::Dense || bDesc.desc.layout != eTensorLayout::Dense )
	{
		logError( u8"Compressed tensors are not currently supported by dbgTensorsDiff() function" );
		return E_NOTIMPL;
	}

	TensorData t0, t1;
	CHECK( t0.initialize( a, aDesc ) );
	CHECK( t1.initialize( b, bDesc ) );
	return TensorData::diff( rdi, t0, t1 );
}