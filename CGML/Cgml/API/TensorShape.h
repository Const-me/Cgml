#pragma once
#include <stdint.h>
#include <array>

namespace Cgml
{
	struct TensorShape
	{
		std::array<uint32_t, 4> size;
		std::array<uint32_t, 4> stride;

		TensorShape();
		TensorShape( const TensorShape& that );
		void operator=( const TensorShape& that );

		__m128i __vectorcall sizeVec() const
		{
			return load( size );
		}
		__m128i __vectorcall stridesVec() const
		{
			return load( stride );
		}
		uint32_t countRows() const
		{
			return size[ 1 ] * size[ 2 ] * size[ 3 ];
		}

		size_t countElements() const
		{
			return horizontalProduct( sizeVec() );
		}

		// True when two tensors have equal count of elements
		static inline bool isSameShape( const TensorShape& t0, const TensorShape& t1 )
		{
			__m128i a = t0.sizeVec();
			__m128i b = t1.sizeVec();
			return vectorEqual( a, b );
		}

		// Reset all fields to zero
		void setZero()
		{
			const __m128i z = _mm_setzero_si128();
			_mm_storeu_si128( ( __m128i* )size.data(), z );
			_mm_storeu_si128( ( __m128i* )stride.data(), z );
		}
	};
}