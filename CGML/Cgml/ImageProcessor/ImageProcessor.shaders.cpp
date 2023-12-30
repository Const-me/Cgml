#include "stdafx.h"
#include "ImageProcessor.h"
#include <Utils/LZ4/lz4.h>

#ifdef DEBUG
#include "Generated/shaders-Debug.inl"
#else
#include "Generated/shaders-Release.inl"
#endif

using namespace Cgml;

HRESULT ImageProcessor::createShaders()
{
	std::vector<uint8_t> dxbc;
	try
	{
		dxbc.resize( shadersUncompressedLength );
	}
	catch( const std::bad_alloc& )
	{
		return E_OUTOFMEMORY;
	}

	const int lz4Status = LZ4_decompress_safe(
		(const char*)s_shadersCompressed.data(),
		(char*)dxbc.data(),
		(int)s_shadersCompressed.size(),
		(int)shadersUncompressedLength );

	if( lz4Status != (int)shadersUncompressedLength )
	{
		logError( u8"LZ4_decompress_safe failed with status %i", lz4Status );
		return PLA_E_CABAPI_FAILURE;
	}

	const uint8_t* const rsi = dxbc.data();
	auto span = spanUpcastTexturePS();
	CHECK( device->CreatePixelShader( rsi + span.first, span.second, nullptr, &psUpcastTexture ) );

	span = spanFullScreenTriangleVS();
	CHECK( device->CreateVertexShader( rsi + span.first, span.second, nullptr, &vsFullScreenTriangle ) );

	span = spanSamplePS();
	CHECK( device->CreatePixelShader( rsi + span.first, span.second, nullptr, &psSample ) );

	span = spanCopyAndNormalizeCS();
	CHECK( device->CreateComputeShader( rsi + span.first, span.second, nullptr, &csCopyAndNormalize ) );

	return S_OK;
}