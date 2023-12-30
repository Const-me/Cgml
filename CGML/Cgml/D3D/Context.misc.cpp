#include "stdafx.h"
#include "Context.h"
using namespace Cgml;

HRESULT COMLIGHTCALL Context::createComputeShaders( int count, const std::pair<int, int>* blobs, const uint8_t* data, int dataSize ) noexcept
{
	if( count < 0 || count > 0xFFFF )
		return E_INVALIDARG;

	shaders.clear();
	if( count == 0 )
		return S_FALSE;

	if( !device )
		return OLE_E_BLANK;
	if( nullptr == blobs || nullptr == data )
		return E_POINTER;
	if( dataSize <= 0 )
		return E_INVALIDARG;

	try
	{
		shaders.resize( count );
	}
	catch( std::bad_alloc& )
	{
		return E_OUTOFMEMORY;
	}

	for( int i = 0; i < count; i++ )
	{
		const std::pair<int, int>& offsets = blobs[ i ];
		const int begin = offsets.first;
		const int end = offsets.second;
		if( begin < 0 || end <= begin || end > dataSize )
			return E_BOUNDS;

		const uint8_t* const rsi = data + begin;
		HRESULT hr = device->CreateComputeShader( rsi, (size_t)( end - begin ), nullptr, &shaders[ i ] );
		CHECK( hr );
	}

	return S_OK;
}

HRESULT COMLIGHTCALL Context::loadImage( iTensor* result, const sImageProcessorParams& ipp, ComLight::iReadStream* stream, uint32_t* previewPixels ) noexcept
{
	std::unique_ptr<iImageProcessor> imageProcessor;
	CHECK( iImageProcessor::create( imageProcessor, device, context, constantBuffers ) );
	return imageProcessor->loadImage( result, ipp, stream, previewPixels );
}