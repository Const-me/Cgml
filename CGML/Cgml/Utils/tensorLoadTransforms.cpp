#include "stdafx.h"
#include "tensorLoadTransforms.h"
#include "Compression/bcml1.h"

namespace
{
	__forceinline void makeIeeeFp16( uint16_t* pointer )
	{
		__m128i* const p = ( __m128i* )pointer;
		__m128i iv = _mm_loadu_si128( p );

		__m256i integers = _mm256_cvtepu16_epi32( iv );
		integers = _mm256_slli_epi32( integers, 16 );

		__m256 floats = _mm256_castsi256_ps( integers );
		iv = _mm256_cvtps_ph( floats, _MM_FROUND_TO_NEAREST_INT );

		_mm_storeu_si128( p, iv );
	}

	static void __declspec( noinline ) makeIeeeFp16( uint16_t* buffer, size_t length )
	{
		constexpr size_t maskAlign8 = ~(size_t)7;
		uint16_t* const endAligned = buffer + ( length & maskAlign8 );
		const size_t rem = length % 8;

		for( ; buffer < endAligned; buffer += 8 )
			makeIeeeFp16( buffer );

		if( 0 == rem )
			return;

		// AVX ISA does not have maskload / maskstore instructions for 2-byte lanes
		// There are only 4-byte vpmaskmovd/vmaskmovps and 8-byte vpmaskmovq/vmaskmovpd
		// That's why the roundtrip pointer -> stack -> registers -> stack -> pointer
		uint16_t remainder[ 8 ];
		_mm_storeu_si128( ( __m128i* )( &remainder[ 0 ] ), _mm_setzero_si128() );
		__movsw( remainder, buffer, rem );
		makeIeeeFp16( remainder );
		__movsw( buffer, remainder, rem );
	}

	static void __declspec( noinline ) downcastFp32Floats( void* pv, size_t length )
	{
		const float* rsi = (const float*)pv;
		uint16_t* rdi = (uint16_t*)pv;

		constexpr size_t maskAlign8 = ~(size_t)7;
		const float* const rsiEndAligned = rsi + ( length & maskAlign8 );
		const size_t rem = length % 8;

		while( rsi < rsiEndAligned )
		{
			__m256 floats = _mm256_loadu_ps( rsi );
			rsi += 8;
			__m128i f16 = _mm256_cvtps_ph( floats, _MM_FROUND_TO_NEAREST_INT );
			_mm_storeu_si128( ( __m128i* )rdi, f16 );
			rdi += 8;
		}

		if( 0 != rem )
		{
			float sourceBuffer[ 8 ];
			uint16_t resultBuffer[ 8 ];
			__movsd( (DWORD*)( &sourceBuffer[ 0 ] ), (const DWORD*)( rsi ), rem );
			__stosd( (DWORD*)( &sourceBuffer[ rem ] ), 0, 8 - rem );

			__m256 floats = _mm256_loadu_ps( rsi );
			__m128i f16 = _mm256_cvtps_ph( floats, _MM_FROUND_TO_NEAREST_INT );
			_mm_storeu_si128( ( __m128i* )( &resultBuffer[ 0 ] ), f16 );
			__movsw( rdi, &resultBuffer[ 0 ], rem );
		}
	}
}

HRESULT Cgml::loadTransform( eLoadTransform tform, eDataType& dt, DXGI_FORMAT& viewFormat, void* pv, size_t& bytes, size_t elements )
{
	assert( tform != eLoadTransform::None );
	if( tform == eLoadTransform::Fp16MakeIeee )
	{
		if( dt != eDataType::BF16 )
			return S_OK;

		if( Bcml1::checkExtensionFlags( Bcml1::eCpuExtensionFlags::AVX2 | Bcml1::eCpuExtensionFlags::F16C ) )
		{
			makeIeeeFp16( (uint16_t*)pv, elements );
			dt = eDataType::FP16;
			viewFormat = DXGI_FORMAT_R16_FLOAT;
			return S_OK;
		}
		else
		{
			logError( u8"The specified tensor format transformation requires AVX2 and F16C support" );
			return HRESULT_FROM_WIN32( ERROR_HV_CPUID_FEATURE_VALIDATION );
		}
	}
	if( tform == eLoadTransform::Fp32DowncastIeee )
	{
		if( dt != eDataType::FP32 )
			return S_OK;
		if( Bcml1::checkExtensionFlags( Bcml1::eCpuExtensionFlags::F16C ) )
		{
			downcastFp32Floats( pv, elements );
			dt = eDataType::FP16;
			viewFormat = DXGI_FORMAT_R16_FLOAT;
			bytes /= 2;
			return S_OK;
		}
		else
		{
			logError( u8"The specified tensor format transformation requires F16C support" );
			return HRESULT_FROM_WIN32( ERROR_HV_CPUID_FEATURE_VALIDATION );
		}
	}
	return E_NOTIMPL;
}