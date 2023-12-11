// This source code is deprecated, and should be dropped by linker.
#include "stdafx.h"
#include "Context.h"
#include <algorithm>
#include <random>
using namespace Cgml;

namespace
{
	HRESULT validateInputs( const sTensorDesc& src, const sTensorDesc& res )
	{
		if( src.dataType != eDataType::FP16 || res.dataType != eDataType::U32 )
			return E_INVALIDARG;

		__m128i v = src.shape.sizeVec();
		v = _mm_srli_si128( v, 4 );
		v = _mm_insert_epi32( v, 1, 3 );
		if( !vectorEqual( v, res.shape.sizeVec() ) )
			return E_INVALIDARG;

		return S_OK;
	}

	// My CPU has 512 KB L2 cache per core.
	// At 8 bytes/elements, 32000 of these items consuming 250 kb RAM, should fit in L2 just fine
	// Possible to reduce by half, with FP16 values and uint16_t indices.
	struct alignas( 8 ) SampleItem
	{
		float f;
		uint32_t idx;

		// Comparisin predicate to sort in descensing order with std::sort
		bool operator<( const SampleItem& that )
		{
			if( f < that.f )
				return false;
			else if( f > that.f )
				return true;
			else
				return idx < that.idx;
		}

		// std::discrete_distribution needs argument type convertable to FP64
		operator double() const { return f; }
	};

	// Store 8 elements, interleaving FP32 values with int32 indices
	__forceinline void store8( SampleItem* rdi, __m256 f, __m256i i )
	{
		__m256i vals = _mm256_castps_si256( f );

		// f0, i0, f1, i1, f4, i4, f5, i5
		__m256i low = _mm256_unpacklo_epi32( vals, i );
		// f2, i2, f3, i3, f6, i6, f7, i7
		__m256i high = _mm256_unpackhi_epi32( vals, i );

		__m256i res = _mm256_inserti128_si256( low, _mm256_castsi256_si128( high ), 1 );
		_mm256_storeu_si256( ( __m256i* )rdi, res );

		res = _mm256_permute2x128_si256( low, high, 0x31 );
		_mm256_storeu_si256( ( __m256i* )( rdi + 4 ), res );
	}

	// Upcast elements FP16 -> FP32, and generate array of SampleItem structures with 0-based element index
	void upcastFloats( SampleItem* rdi, size_t length, const uint16_t* rsi )
	{
		constexpr size_t maskAlign8 = ~(size_t)7;
		const uint16_t* rsiEndAligned = rsi + ( length & maskAlign8 );

		__m256i indexVector = _mm256_setr_epi32( 0, 1, 2, 3, 4, 5, 6, 7 );
		while( rsi < rsiEndAligned )
		{
			__m128i iv = _mm_loadu_si128( ( const __m128i* )rsi );
			rsi += 8;
			__m256 upcasted8 = _mm256_cvtph_ps( iv );
			store8( rdi, upcasted8, indexVector );
			rdi += 8;
			indexVector = _mm256_add_epi32( indexVector, _mm256_set1_epi32( 8 ) );
		}

		const size_t rem = length % 8;
		if( 0 != rem )
		{
			uint16_t bufferIn[ 8 ];
			_mm_storeu_si128( ( __m128i* )bufferIn, _mm_setzero_si128() );
			__movsw( bufferIn, rsi, rem );

			__m128i iv = _mm_loadu_si128( ( const __m128i* )bufferIn );
			__m256 upcasted8 = _mm256_cvtph_ps( iv );
			SampleItem bufferOut[ 8 ];
			store8( bufferOut, upcasted8, indexVector );
			__movsq( (uint64_t*)rdi, (const uint64_t*)bufferOut, rem );
		}
	}

	class TopPImpl
	{
		iContext& context;
		iTensor* const dest;
		iTensor* const source;
		const float prob;

		sTensorDesc descResult, descSource;

		// Temporary data for 1 row being processed
		std::vector<SampleItem> rowTemp;

		// Output tensor only contains a few integers, no need to mess with the streaming.
		std::vector<uint32_t> result;

		HRESULT consumeMappedData( const uint16_t* rsi, size_t countElements );

		static HRESULT __stdcall readCallbackStatic( const void* rsi, uint32_t cb, void* pv )
		{
			if( 0 != ( cb % 2 ) )
				return E_UNEXPECTED;
			TopPImpl* tpi = (TopPImpl*)pv;
			return tpi->consumeMappedData( (const uint16_t*)rsi, cb / 2 );
		}

		HRESULT writeTensorData( uint32_t* rdi, size_t countElements ) const;

		static HRESULT __stdcall writeCallbackStatic( void* rdi, uint32_t cb, void* pv )
		{
			if( 0 != ( cb % 4 ) )
				return E_UNEXPECTED;
			TopPImpl* tpi = (TopPImpl*)pv;
			return tpi->writeTensorData( (uint32_t*)rdi, cb / 4 );
		}

	public:
		TopPImpl( iContext* c, iTensor* d, iTensor* s, float p ) :
			context( *c ), dest( d ), source( s ), prob( p )
		{ }

		HRESULT sample();
	};

	HRESULT TopPImpl::sample()
	{
		if( nullptr == dest || nullptr == source )
			return E_POINTER;

		source->getDesc( descSource );
		dest->getDesc( descResult );
		CHECK( validateInputs( descSource, descResult ) );

		try
		{
			result.resize( descResult.shape.countElements() );
			const size_t rowLength = descSource.shape.size[ 0 ];
			rowTemp.resize( rowLength );
		}
		catch( const std::bad_alloc& )
		{
			return E_OUTOFMEMORY;
		}

		CHECK( context.download( source, &TopPImpl::readCallbackStatic, this, Cgml::eDownloadFlag::None ) );

		CHECK( context.writeDynamic( dest, descResult.shape, &TopPImpl::writeCallbackStatic, this ) );

		return S_OK;
	}

	HRESULT TopPImpl::consumeMappedData( const uint16_t* rsi, size_t countElements )
	{
		const size_t rowLength = descSource.shape.size[ 0 ];
		std::random_device rd;
		std::mt19937 gen( rd() );

		for( uint32_t& rdi : result )
		{
			upcastFloats( rowTemp.data(), rowLength, rsi );
			rsi += rowLength;

			// Sort in descending order, see `operator <` of that thing
			std::sort( rowTemp.begin(), rowTemp.end() );

			// Find initial slice of the vector, where cumulative sum of these probabilities doesn't exceed the parameter
			double cumSum = 0;
			size_t headEnd = 0;
			for( ; headEnd < rowLength; headEnd++ )
			{
				double newSum = cumSum + rowTemp[ headEnd ].f;
				if( newSum > prob )
					break;
				cumSum = newSum;
			}

			if( headEnd > 1 )
			{
				// We have at least 2 elements to pick
				std::discrete_distribution<uint32_t> dist( rowTemp.begin(), rowTemp.begin() + headEnd );
				const uint32_t idx = dist( gen );
				rdi = rowTemp[ idx ].idx;
			}
			else
			{
				// There's nothing wring with this branch, it simply means the model is pretty confident in what's next.
				rdi = rowTemp[ 0 ].idx;
			}
		}
		return S_OK;
	}

	HRESULT TopPImpl::writeTensorData( uint32_t* rdi, size_t countElements ) const
	{
		if( countElements != result.size() )
			return E_UNEXPECTED;
		__movsd( (DWORD*)rdi, (const DWORD*)result.data(), countElements );
		return S_OK;
	}
}

#if false
HRESULT COMLIGHTCALL Context::sampleTopP( iTensor* dest, iTensor* source, float p ) noexcept
{
	TopPImpl impl{ this, dest, source, p };
	return impl.sample();
}
#endif