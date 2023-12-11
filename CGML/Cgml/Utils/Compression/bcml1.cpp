#include "stdafx.h"
#include "bcml1.h"

namespace
{
	__forceinline __m256 loadfp16( const uint16_t* rsi )
	{
		__m128i v = _mm_loadu_si128( ( const __m128i* )rsi );
		return _mm256_cvtph_ps( v );
	}
	__forceinline __m256 loadbf16( const uint16_t* rsi )
	{
		__m128i v = _mm_loadu_si128( ( const __m128i* )rsi );
		__m256i ext = _mm256_cvtepu16_epi32( v );
		ext = _mm256_slli_epi32( ext, 16 );
		return _mm256_castsi256_ps( ext );
	}

	__forceinline void storeHeaderFp16( uint32_t* rdi, __m128 floats )
	{
		__m128i header16 = _mm_cvtps_ph( floats, _MM_FROUND_NINT );
		_mm_storeu_si32( rdi, header16 );
	}

	__forceinline void storeHeaderBf16( uint32_t* rdi, __m128 floats )
	{
		__m128i iv = _mm_castps_si128( floats );

		// Scalar version:
		// uint32_t rounding_bias = ((U32 >> 16) & 1) + UINT32_C(0x7FFF);
		// output_row[ col ] = static_cast<uint16_t>( ( U32 + rounding_bias ) >> 16 );
		__m128i bias = _mm_srli_epi32( iv, 16 );
		bias = _mm_and_si128( bias, _mm_set1_epi32( 1 ) );
		bias = _mm_add_epi32( bias, _mm_set1_epi32( 0x7fff ) );
		iv = _mm_add_epi32( iv, bias );
		iv = _mm_srli_epi32( iv, 16 );
		iv = _mm_packus_epi32( iv, iv );
		_mm_storeu_si32( rdi, iv );
	}

	__forceinline __m128 horizontalMin( __m256 v8 )
	{
		__m128 v = _mm256_extractf128_ps( v8, 1 );
		v = _mm_min_ps( v, _mm256_castps256_ps128( v8 ) );
		v = _mm_min_ps( v, _mm_movehl_ps( v, v ) );
		v = _mm_min_ss( v, _mm_movehdup_ps( v ) );
		return v;
	}

	__forceinline __m128 horizontalMax( __m256 v8 )
	{
		__m128 v = _mm256_extractf128_ps( v8, 1 );
		v = _mm_max_ps( v, _mm256_castps256_ps128( v8 ) );
		v = _mm_max_ps( v, _mm_movehl_ps( v, v ) );
		v = _mm_max_ss( v, _mm_movehdup_ps( v ) );
		return v;
	}

	inline __m128i packNibbles( __m256i bytes )
	{
		// Move bits within 16-bit lanes from 0000_abcd_0000_efgh into 0000_0000_abcd_efgh
		const __m256i lowByte = _mm256_set1_epi16( 0xFF );
		__m256i high = _mm256_andnot_si256( lowByte, bytes );
		__m256i low = _mm256_and_si256( lowByte, bytes );
		high = _mm256_srli_epi16( high, 4 );
		bytes = _mm256_or_si256( low, high );

		// Compress uint16_t lanes into bytes
		__m128i r0 = _mm256_castsi256_si128( bytes );
		__m128i r1 = _mm256_extracti128_si256( bytes, 1 );
		return _mm_packus_epi16( r0, r1 );
	}

	// Load 32 FP16 values, quantize to specific bit depth, store block header,
	// and return a vector of the weights, 4 bits/element
	template<uint8_t outputBits, bool bf16Source, bool bf16Header>
	__forceinline __m128i quantizeBlock32( uint32_t* rdi, const uint16_t* rsi )
	{
		// Load elements into 4 AVX vectors
		__m256 v0, v1, v2, v3;
		if constexpr( bf16Source )
		{
			v0 = loadbf16( rsi );
			v1 = loadbf16( rsi + 8 );
			v2 = loadbf16( rsi + 16 );
			v3 = loadbf16( rsi + 24 );
		}
		else
		{
			v0 = loadfp16( rsi );
			v1 = loadfp16( rsi + 8 );
			v2 = loadfp16( rsi + 16 );
			v3 = loadfp16( rsi + 24 );
		}

		// Compute min and max for the block
		__m256 min = v0;
		__m256 max = v0;
		min = _mm256_min_ps( min, v1 );
		max = _mm256_max_ps( max, v1 );
		min = _mm256_min_ps( min, v2 );
		max = _mm256_max_ps( max, v2 );
		min = _mm256_min_ps( min, v3 );
		max = _mm256_max_ps( max, v3 );

		__m128 min1 = horizontalMin( min );
		__m128 max1 = horizontalMax( max );

		// Offset all 32 numbers relative to the minimum
		min = _mm256_broadcastss_ps( min1 );
		v0 = _mm256_sub_ps( v0, min );
		v1 = _mm256_sub_ps( v1, min );
		v2 = _mm256_sub_ps( v2, min );
		v3 = _mm256_sub_ps( v3, min );

		// These HLSL shaders are decompressing them like that:
		// f32[i] = mad( scale, i4[i], offset ), where i4 integers have 3 or 4 bits,
		// i.e. the numbers are in [ 0 .. 7 ] or [ 0 .. 15 ] interval
		// offset = min, scale = ( max - min ) / 15
		// And then, to compute these integers, we need the inverse multiplier, 15.0 / ( max - min )
		const __m128 dist = _mm_moveldup_ps( _mm_sub_ss( max1, min1 ) );

		constexpr float scalingScalar = ( 1 << outputBits ) - 1;
		const __m128 scaling = _mm_set1_ps( scalingScalar );

		// dist, 15, 15, 15
		__m128 t0 = _mm_blend_ps( dist, scaling, 0b1110 );

		// 15, dist, 15, 15
		__m128 t1 = _mm_blend_ps( dist, scaling, 0b1101 );

		// [ dist / 15, 15 / dist, 1.0, 1.0 ]
		__m128 divs = _mm_div_ps( t0, t1 );

		// Store two FP16 numbers in the block header, they are [ dist / 15, min ]
		__m128 headerFloats = _mm_unpacklo_ps( divs, min1 );
		if constexpr( bf16Header )
			storeHeaderBf16( rdi, headerFloats );
		else
			storeHeaderFp16( rdi, headerFloats );

		// Compute the following expression without branches:
		// divs.x = ( divs.x > 0 ) ? divs.y : 0.0
		const __m128 notEqual = _mm_cmpgt_ss( divs, _mm_setzero_ps() );
		divs = _mm_movehdup_ps( divs );
		divs = _mm_and_ps( divs, notEqual );

		// Apply the multiplier
		__m256 mul8 = _mm256_broadcastss_ps( divs );
		v0 = _mm256_mul_ps( v0, mul8 );
		v1 = _mm256_mul_ps( v1, mul8 );
		v2 = _mm256_mul_ps( v2, mul8 );
		v3 = _mm256_mul_ps( v3, mul8 );

		// Round to nearest integer
		v0 = _mm256_round_ps( v0, _MM_ROUND_NEAREST );
		v1 = _mm256_round_ps( v1, _MM_ROUND_NEAREST );
		v2 = _mm256_round_ps( v2, _MM_ROUND_NEAREST );
		v3 = _mm256_round_ps( v3, _MM_ROUND_NEAREST );

		// Convert floats to integers
		__m256i i0 = _mm256_cvtps_epi32( v0 );
		__m256i i1 = _mm256_cvtps_epi32( v1 );
		__m256i i2 = _mm256_cvtps_epi32( v2 );
		__m256i i3 = _mm256_cvtps_epi32( v3 );

		// Convert int32 to int16
		i0 = _mm256_packs_epi32( i0, i1 );
		i2 = _mm256_packs_epi32( i2, i3 );
		// Convert int16 to uint8
		i0 = _mm256_packus_epi16( i0, i2 );
		// Clamp into [ 0 .. 15 ] or [ 0 .. 7 ], to compensate for possible numerical issues with the floating-point math
		// The unsigned saturation of _mm256_packus_epi16 already clamped to x >= 0 so we only need to enforce the upper bound

		constexpr int maximumInteger = ( 1 << outputBits ) - 1;
		i0 = _mm256_min_epu8( i0, _mm256_set1_epi8( (int8_t)maximumInteger ) );

		// Compress the vector into 4 bit/value
		__m128i res = packNibbles( i0 );

		// The AVX2 pack instructions above process 16-byte pieces independently
		// For this reason, the order of the values is now wrong, the following shuffle instruction is fixing that
		// vpshufb shuffles 16-bytes vectors, 3 times faster than vpermd which shuffles across the complete 32-bytes vectors
		const __m128i perm = _mm_setr_epi8( 0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15 );
		res = _mm_shuffle_epi8( res, perm );

		return res;
	}

	inline void reshapePanel( uint32_t* rdi, const uint32_t* rsi, size_t widthIntegers )
	{
		constexpr size_t panelHeight = Bcml1::PANEL_HEIGHT;
		uint32_t* const rdiEnd = rdi + panelHeight;
		for( ; rdi < rdiEnd; rdi++ )
		{
			uint32_t* rdiRow = rdi;
			const uint32_t* const rsiEnd = rsi + widthIntegers;
			for( ; rsi < rsiEnd; rsi++, rdiRow += panelHeight )
				*rdiRow = *rsi;
		}
	}

	// Load FP16 elements, quantize to 4 bits, produce BCML1 compressed tensor
	struct BCML1_F16
	{
		static constexpr size_t blockWidth = 32;
		static constexpr size_t integersPerBlock = 5;

		static __forceinline void compressBlock( uint32_t* rdi, const uint16_t* rsi )
		{
			const __m128i weights = quantizeBlock32<4, false, false>( rdi, rsi );
			_mm_storeu_si128( ( __m128i* )( rdi + 1 ), weights );
		}
	};

	// Load BF16 elements, quantize to 4 bits, produce BCML1 compressed tensor
	struct BCML1_BF16
	{
		static constexpr size_t blockWidth = 32;
		static constexpr size_t integersPerBlock = 5;

		static __forceinline void compressBlock( uint32_t* rdi, const uint16_t* rsi )
		{
			const __m128i weights = quantizeBlock32<4, true, false>( rdi, rsi );
			_mm_storeu_si128( ( __m128i* )( rdi + 1 ), weights );
		}
	};

	struct BCML1E
	{
		static constexpr size_t blockWidth = 32;
		static constexpr size_t integersPerBlock = 5;

		static __forceinline void compressBlock( uint32_t* rdi, const uint16_t* rsi )
		{
			const __m128i weights = quantizeBlock32<4, true, true>( rdi, rsi );
			_mm_storeu_si128( ( __m128i* )( rdi + 1 ), weights );
		}
	};

	struct BCML2
	{
		static constexpr size_t blockWidth = 32;
		static constexpr size_t integersPerBlock = 4;

		static __forceinline void compressBlock( uint32_t* rdi, const uint16_t* rsi )
		{
			const __m128i weights = quantizeBlock32<3, false, false>( rdi, rsi );

			// Move the block payload from the vector to 2 scalar registers
			uint64_t high = (uint64_t)_mm_extract_epi64( weights, 1 );
			uint64_t low = (uint64_t)_mm_cvtsi128_si64( weights );
			assert( 0 == ( ( high | low ) & 0x8888888888888888ull ) );

			// Gather these precious bits with BMI2 instructions
			constexpr uint64_t gatherBits = 0x7777777777777777ull;
			high = _pext_u64( high, gatherBits );
			low = _pext_u64( low, gatherBits );

			// Now we have 2 values, each containing 48 bits = 6 bytes of the payload

			// Store initial 4 bytes of the payload
			rdi[ 1 ] = (uint32_t)low;

			// Store the remaining 8 bytes of the payload
			low >>= 32;
			high <<= 16;
			high |= low;
			*(uint64_t*)( rdi + 2 ) = high;
		}
	};

	using Bcml1::PANEL_HEIGHT;

	template<class Codec>
	static HRESULT compressImpl( const Cgml::sTensorDesc& desc, const std::vector<__m256i>& sourceVector, std::vector<uint32_t>& result )
	{
		size_t compressedBytes = desc.shape.stride[ 3 ];
		compressedBytes *= desc.shape.size[ 3 ];
		assert( 0 == compressedBytes % 4 );

		size_t compressedElts = compressedBytes / 4;
		try
		{
			result.resize( compressedElts );
		}
		catch( const std::bad_alloc& )
		{
			return E_OUTOFMEMORY;
		}

		const size_t width = desc.shape.size[ 0 ];
		const size_t completeBlocks = width / 32;
		const size_t remainder = width % 32;

		const size_t countRows = desc.shape.size[ 1 ] * desc.shape.size[ 2 ] * desc.shape.size[ 3 ];

		const uint16_t* rsi = (const uint16_t*)sourceVector.data();
		uint32_t* rdi = result.data();

		for( size_t i = 0; i < countRows; i++ )
		{
			for( size_t j = 0; j < completeBlocks; j++ )
			{
				Codec::compressBlock( rdi, rsi );
				rdi += Codec::integersPerBlock;
				rsi += 32;
			}

			if( 0 != remainder )
			{
				uint16_t bufferIn[ 32 ];
				__movsw( bufferIn, rsi, remainder );
				// For optimal compression quality, pad incomplete blocks with the last value
				__stosw( bufferIn + remainder, rsi[ remainder - 1 ], 32 - remainder );

				Codec::compressBlock( rdi, rsi );
				rdi += Codec::integersPerBlock;
				rsi += remainder;
			}
		}

		// We have compressed the source data in row major layout
		// Need one last step - reshape these integers into panels

		const size_t widthBlocks = ( width + 31 ) / 32;
		const size_t widthIntegers = widthBlocks * Codec::integersPerBlock;
		const size_t panelsCount = ( desc.shape.size[ 1 ] + PANEL_HEIGHT - 1 ) / PANEL_HEIGHT;
		const size_t completePanels = desc.shape.size[ 1 ] / PANEL_HEIGHT;
		const size_t panelIntegers = widthIntegers * PANEL_HEIGHT;
		std::vector<uint32_t> panelTemp;
		try
		{
			panelTemp.resize( panelIntegers );
		}
		catch( const std::bad_alloc& )
		{
			return E_OUTOFMEMORY;
		}

		// TODO [low]: optimize this into AVX2, loading/storing vectors instead of scalars, 
		// using _mm256_inserti128_si256, _mm256_unpacklo_epi32, _mm256_unpackhi_epi32, etc.
		rdi = result.data();
		for( uint32_t* const rdiEnd = rdi + completePanels * panelIntegers; rdi < rdiEnd; rdi += panelIntegers )
		{
			// Copy row-major data into temporary buffer
			__movsd( (DWORD*)panelTemp.data(), (const DWORD*)rdi, panelIntegers );

			// Copy transposed panel from temporary buffer back to the destination
			reshapePanel( rdi, panelTemp.data(), widthIntegers );
		}

		if( completePanels != panelsCount )
		{
			// Tensor height is not a multiple of PANEL_HEIGHT, need the the final incomplete panel
			const size_t remainderRows = desc.shape.size[ 1 ] % PANEL_HEIGHT;
			const size_t remainderIntegers = remainderRows * widthIntegers;

			// Copy incomplete rows to temporary vector
			__movsd( (DWORD*)panelTemp.data(), (const DWORD*)rdi, remainderIntegers );
			// Fill the rest with zeros
			__stosd( (DWORD*)panelTemp.data() + remainderIntegers, 0, panelIntegers - remainderIntegers );

			// Copy transposed panel from temporary buffer back to the destination
			reshapePanel( rdi, panelTemp.data(), widthIntegers );
		}

		return S_OK;
	}

	template<class Codec>
	uint32_t rowWidthBytes( uint32_t elements )
	{
		uint32_t blocks = ( elements + Codec::blockWidth - 1 ) / Codec::blockWidth;
		uint32_t integers = blocks * Codec::integersPerBlock;
		return integers * 4;
	}

	static uint8_t collectCpuidBits()
	{
		using Bcml1::eCpuExtensionFlags;
		uint8_t res = checkAvx2Support() ? (uint8_t)eCpuExtensionFlags::AVX2 : 0;
		res |= checkF16cSuppport() ? (uint8_t)eCpuExtensionFlags::F16C : 0;
		res |= checkBmi2Suppport() ? (uint8_t)eCpuExtensionFlags::BMI2 : 0;
		return res;
	}

	static const uint8_t cpuIdBits = collectCpuidBits();
}

bool Bcml1::checkExtensionFlags( eCpuExtensionFlags required )
{
	const uint8_t req = (uint8_t)required;
	return req == ( cpuIdBits & req );
}

HRESULT Bcml1::makeDesc( sTensorDesc& rdi, const sTensorDesc& rsi )
{
	if( rsi.usage != eBufferUse::Immutable )
	{
		logError( u8"BCML compression is only implemented for immutable tensors" );
		return E_NOTIMPL;
	}

	eCpuExtensionFlags requiredFlags = eCpuExtensionFlags::AVX2;
	rdi = rsi;
	if( rsi.dataType != eDataType::FP16 && rsi.dataType != eDataType::BF16 )
	{
		logError( u8"BCML compression is only implemented for FP16 and BF16 inputs" );
		return E_NOTIMPL;
	}
	else
	{
		// We gonna upcast and downcast FP16 numbers, check the HW support
		requiredFlags |= eCpuExtensionFlags::F16C;
	}

	const uint32_t widthElements = rsi.shape.size[ 0 ];
	uint32_t rowBytes;
	switch( rsi.layout )
	{
	case Cgml::eTensorLayout::BCML1:
		rowBytes = rowWidthBytes<BCML1_F16>( widthElements );
		break;
		/* case Cgml::eTensorLayout::BCML2:
			rowBytes = rowWidthBytes<BCML2>( widthElements );
			// That codec uses BMI2 to compress these 3-bit fields
			requiredFlags |= eCpuExtensionFlags::BMI2;
			break; */
	default:
		return E_INVALIDARG;
	}

	if( !checkExtensionFlags( requiredFlags ) )
	{
		logError( u8"BCML compressor requires a CPU with AVX2, F16C, and/or BMI2 support, depending on the codec version" );
		return HRESULT_FROM_WIN32( ERROR_HV_CPUID_FEATURE_VALIDATION );
	}

	const uint32_t panelBytes = rowBytes * PANEL_HEIGHT;

	const uint32_t panelsCount = ( rsi.shape.size[ 1 ] + PANEL_HEIGHT - 1 ) / PANEL_HEIGHT;

	rdi.shape.stride[ 0 ] = 0;
	rdi.shape.stride[ 1 ] = panelBytes;
	uint32_t s = panelBytes * panelsCount;
	rdi.shape.stride[ 2 ] = s;
	s *= rsi.shape.size[ 2 ];
	rdi.shape.stride[ 3 ] = s;

	rdi.dataType = eDataType::U32;
	return S_OK;
}

namespace
{
	constexpr uint16_t makeKey( Cgml::eDataType source, Cgml::eTensorLayout dest )
	{
		uint16_t res = (uint8_t)dest;
		res <<= 8;
		res |= (uint8_t)source;
		return res;
	}
}

HRESULT Bcml1::compress( eDataType sourceType, const sTensorDesc& desc, const std::vector<__m256i>& sourceVector, std::vector<uint32_t>& result )
{
	const uint16_t key = makeKey( sourceType, desc.layout );
	switch( key )
	{
	case makeKey( eDataType::FP16, eTensorLayout::BCML1 ):
		return compressImpl<BCML1_F16>( desc, sourceVector, result );
	case makeKey( eDataType::BF16, eTensorLayout::BCML1 ):
		return compressImpl<BCML1_BF16>( desc, sourceVector, result );
	}
	return E_NOTIMPL;

	/*
	switch( desc.layout )
	{
	case eTensorLayout::BCML1:
		return compressImpl<BCML1>( desc, sourceVector, result );
		case eTensorLayout::BCML1E:
			return compressImpl<BCML1E>( desc, sourceVector, result );
		case eTensorLayout::BCML2:
			return compressImpl<BCML2>( desc, sourceVector, result );
	}
	return E_NOTIMPL;
	*/
}