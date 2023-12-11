#include <stdafx.h>
#include "miscUtils.h"

namespace
{
	__forceinline __m128i downcastAndCompare( __m256 v, __m256& inexact )
	{
		__m128i f16 = _mm256_cvtps_ph( v, _MM_FROUND_TO_NEAREST_INT );

		__m256 upcasted = _mm256_cvtph_ps( f16 );
		__m256 neq = _mm256_cmp_ps( v, upcasted, _CMP_NEQ_OQ );
		inexact = _mm256_or_ps( inexact, neq );
 		return f16;
	}

	__forceinline void isZero( __m256 v, __m256& notZero )
	{
		__m256 neq = _mm256_cmp_ps( v, _mm256_setzero_ps(), _CMP_NEQ_OQ );
		notZero = _mm256_or_ps( notZero, neq );
	}
}

// Downcast F32 to F16 in-place using "round to nearest" mode,
// and return a boolean telling whether the output is exactly equal
// When this function returns true, no data is lost.
BOOL __stdcall downcastFloats( void* buffer, uint32_t length )
{
	__m256 inexact = _mm256_setzero_ps();

	__m128i* rdi = ( __m128i* )buffer;
	const float* rsi = (float*)buffer;
	const float* const rsiEndAligned = rsi + ( length / 8 ) * 8;
	const size_t rem = length % 8;
	while( rsi < rsiEndAligned )
	{
		__m256 v = _mm256_loadu_ps( rsi );
		rsi += 8;

		__m128i f16 = downcastAndCompare( v, inexact );

		_mm_storeu_si128( rdi, f16 );
		rdi++;
	}

	if( rem != 0 )
	{
		__m256i mask = makeAvxMask( rem );
		__m256 v = _mm256_maskload_ps( rsi, mask );

		__m128i f16 = downcastAndCompare( v, inexact );

		WORD remainder[ 8 ];
		_mm_storeu_si128( ( __m128i* )remainder, f16 );
		__movsw( (WORD*)rdi, remainder, rem );
	}

	return _mm256_testz_ps( inexact, inexact ) ? TRUE : FALSE;
}

BOOL __stdcall isAllZero( const float* rsi, uint32_t length )
{
	__m256 notZero = _mm256_setzero_ps();

	const float* const rsiEndAligned = rsi + ( length / 8 ) * 8;
	const size_t rem = length % 8;
	while( rsi < rsiEndAligned )
	{
		__m256 v = _mm256_loadu_ps( rsi );
		rsi += 8;
		isZero( v, notZero );
	}

	if( rem != 0 )
	{
		__m256i mask = makeAvxMask( rem );
		__m256 v = _mm256_maskload_ps( rsi, mask );
		isZero( v, notZero );
	}

	return _mm256_testz_ps( notZero, notZero ) ? TRUE : FALSE;
}