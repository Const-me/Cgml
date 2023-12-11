#include "stdafx.h"
#include "Device.h"
#include "../Utils/LargeBuffer.h"
#include "../Utils/MemoryReader.h"
#include "Tensor.h"
#include "tensorUtils.h"
#include "createBuffer.h"
#include <Utils/tensorLoadTransforms.h>
using namespace Cgml;

HRESULT Device::createTensor( iTensor** pp, const sTensorDesc& desc, iTensor* reuse ) noexcept
{
	if( desc.usage == eBufferUse::Immutable )
	{
		logError( u8"iDevice.createTensor can't create tensors with Immutable usage, they require initial data" );
		return E_INVALIDARG;
	}
	BufferDesc bufferDesc;
	CHECK( bufferDesc.create( desc ) );

	if( nullptr != reuse )
	{
		Tensor* old = static_cast<Tensor*>( reuse );
		if( old->getDesc().usage == eBufferUse::Immutable )
			return E_INVALIDARG;

		ResizeableTensor* rt = static_cast<ResizeableTensor*>( old );
		CHECK( rt->resize( device, desc ) );

		// Note this probably breaks when using from C++, which assumes all output pointers are returned with ref.count = 1
		// If you call this from C++ and using CComPtr<iTensor>, compare input + output COM pointers, and call AddRef() if they're equal.
		// ComLight maintains a global cache of live objects, will return same instance
		*pp = reuse;
		return S_FALSE;
	}

	if( desc.usage == eBufferUse::ReadWrite )
	{
		CComPtr<ID3D11ShaderResourceView> srv;
		CComPtr<ID3D11UnorderedAccessView> uav;

		CHECK( createDefault( device, bufferDesc, srv, uav ) );

		ComLight::CComPtr<ComLight::Object<ResizeableTensor>> result;
		CHECK( ComLight::Object<ResizeableTensor>::create( result, desc, srv, uav ) );
		result.detach( pp );
		return S_OK;
	}
	else if( desc.usage == eBufferUse::ReadWriteDownload )
	{
		CComPtr<ID3D11ShaderResourceView> srv;
		CComPtr<ID3D11UnorderedAccessView> uav;
		CComPtr<ID3D11Buffer> vram, staging;

		CHECK( createStaging( device, bufferDesc, srv, uav, vram, staging ) );

		ComLight::CComPtr<ComLight::Object<StagingTensor>> result;
		CHECK( ComLight::Object<StagingTensor>::create( result, desc, srv, uav, vram, staging ) );
		result.detach( pp );
		return S_OK;
	}
	else if( desc.usage == eBufferUse::Dynamic )
	{
		CComPtr<ID3D11ShaderResourceView> srv;
		CComPtr<ID3D11Buffer> vram;

		CHECK( createDynamic( device, bufferDesc, srv, vram ) );

		ComLight::CComPtr<ComLight::Object<DynamicTensor>> result;
		CHECK( ComLight::Object<DynamicTensor>::create( result, desc, srv, vram ) );
		result.detach( pp );
		return S_OK;
	}
	else
		return E_UNEXPECTED;
}

HRESULT Device::loadImmutableTensor( iTensor** pp, const sTensorDesc& desc, ComLight::iReadStream* stream, uint32_t length, eLoadTransform tform ) noexcept
{
	if( nullptr == pp || nullptr == stream )
		return E_POINTER;

	if( desc.usage != eBufferUse::Immutable )
	{
		logError( u8"iDevice.uploadImmutableTensor can only create tensors with eBufferUse.Immutable" );
		return E_INVALIDARG;
	}

	if( desc.layout != eTensorLayout::Dense )
		return loadCompressed( pp, desc, stream, length );

	const size_t elements = horizontalProduct( desc.shape.sizeVec() );
	const size_t elementsPadding = (size_t)desc.shape.stride[ 3 ] * desc.shape.size[ 3 ];
	if( elements != elementsPadding )
	{
		logError( u8"The input data is expected to be dense, i.e. no padding" );
		return E_INVALIDARG;
	}

	if( 0 != ( elements >> 32 ) )
	{
		logError( u8"The tensor is too large, exceeds 4G elements" );
		return DISP_E_OVERFLOW;
	}

	sTypeInfo ti;
	CHECK( sTypeInfo::initialize( ti, desc ) );

	const size_t bufferBytes = elements * ti.cbElement;
	if( 0 != ( bufferBytes >> 31 ) )
	{
		logError( u8"The tensor is too large, exceeds 2GB VRAM" );
		return DISP_E_OVERFLOW;
	}

	if( bufferBytes > length )
	{
		logError( u8"Tensor length does not match" );
		return E_FAIL;
	}

	LargeBuffer buffer;
	CHECK( buffer.allocate( length ) );
	CHECK( stream->read( buffer.pointer(), length ) );

	if( tform == eLoadTransform::None )
		return uploadImmutable( pp, desc, buffer.pointer(), bufferBytes, ti.format, (UINT)elements );
	else
	{
		sTensorDesc d2 = desc;
		CHECK( Cgml::loadTransform( tform, d2.dataType, ti.format, buffer.pointer(), elements ) );
		return uploadImmutable( pp, d2, buffer.pointer(), bufferBytes, ti.format, (UINT)elements );
	}
}

HRESULT Device::uploadImmutableTensor( iTensor** pp, const sTensorDesc& desc, const void* rsi, uint32_t length ) noexcept
{
	if( nullptr == pp || nullptr == rsi )
		return E_POINTER;

	if( desc.layout != eTensorLayout::Dense )
	{
		MemoryReader reader( rsi, length );
		return loadCompressed( pp, desc, &reader, length );
	}

	const size_t elements = horizontalProduct( desc.shape.sizeVec() );
	const size_t elementsPadding = (size_t)desc.shape.stride[ 3 ] * desc.shape.size[ 3 ];
	if( elements != elementsPadding )
	{
		logError( u8"The input data is expected to be dense, i.e. no padding" );
		return E_INVALIDARG;
	}

	sTypeInfo ti;
	CHECK( sTypeInfo::initialize( ti, desc ) );

	const size_t bufferBytes = elements * ti.cbElement;
	if( 0 != ( bufferBytes >> 31 ) )
	{
		logError( u8"The tensor is too large, exceeds 2GB VRAM" );
		return DISP_E_OVERFLOW;
	}
	if( bufferBytes != length )
	{
		logError( u8"Unexpected payload length" );
		return E_INVALIDARG;
	}

	return uploadImmutable( pp, desc, rsi, bufferBytes, ti.format, (UINT)elements );
}

HRESULT Device::uploadImmutable( iTensor** pp, const sTensorDesc& desc, const void* rsi, size_t length, DXGI_FORMAT format, UINT bufferElements )
{
	CComPtr<ID3D11ShaderResourceView> srv;
	createImmutableBuffer( device, rsi, length, bufferElements, format, srv );

	ComLight::CComPtr<ComLight::Object<Tensor>> result;
	CHECK( ComLight::Object<Tensor>::create( result, desc, srv ) );
	result.detach( pp );
	return S_OK;
}

static HRESULT queryOptionalFeatures( ID3D11Device* dev, uint8_t& rdi )
{
	rdi = 0;

	D3D11_FEATURE_DATA_DOUBLES fdDoubles;
	HRESULT hr = dev->CheckFeatureSupport( D3D11_FEATURE_DOUBLES, &fdDoubles, sizeof( fdDoubles ) );
	if( SUCCEEDED( hr ) && fdDoubles.DoublePrecisionFloatShaderOps )
	{
		rdi |= 1;	// eOptionalFeatures.FP64Basic

		D3D11_FEATURE_DATA_D3D11_OPTIONS fdOptions;
		hr = dev->CheckFeatureSupport( D3D11_FEATURE_D3D11_OPTIONS, &fdOptions, sizeof( fdOptions ) );
		if( SUCCEEDED( hr ) && fdOptions.ExtendedDoublesShaderInstructions )
			rdi |= 2;	// eOptionalFeatures.FP64Advanced
	}
	return S_OK;
}

HRESULT Device::getDeviceInfo( sDeviceInfo& rdi ) noexcept
{
	if( !device )
		return OLE_E_BLANK;

	CComPtr<IDXGIDevice> dxgi;
	CHECK( device->QueryInterface( &dxgi ) );

	CComPtr<IDXGIAdapter> gpu;
	CHECK( dxgi->GetAdapter( &gpu ) );

	DXGI_ADAPTER_DESC desc;
	CHECK( gpu->GetDesc( &desc ) );

	// Copy the name from stack to the class field
	size_t len = wcsnlen_s( desc.Description, 128 );
	deviceName.assign( desc.Description, desc.Description + len );
	rdi.name = deviceName.c_str();

	// The rest of the fields in the output structure are scalar values, no need to make copies
	rdi.vram = desc.DedicatedVideoMemory;
	rdi.vendor = (uint16_t)desc.VendorId;

	uint16_t fl = device->GetFeatureLevel();
	rdi.featureLevelMajor = (uint8_t)( ( fl >> 12 ) & 0xF );
	rdi.featureLevelMinor = (uint8_t)( ( fl >> 8 ) & 0xF );

	queryOptionalFeatures( device, rdi.optionalFeatures );
	return S_OK;
}

HRESULT Device::waitForWeightsCompressor() noexcept
{
	if( !weightCompressor )
		return S_OK;

	CHECK( weightCompressor->join() );
	weightCompressor.reset();
	return S_OK;
}

HRESULT Device::loadCompressed( iTensor** pp, const sTensorDesc& desc, ComLight::iReadStream* stream, uint32_t length ) noexcept
{
	if( desc.shape.stride[ 0 ] != 0 )
	{
		// The input is dense uncompressed tensor; compress with AVX2, on background threads
		const size_t elements = horizontalProduct( desc.shape.sizeVec() );
		const size_t elementsPadding = (size_t)desc.shape.stride[ 3 ] * desc.shape.size[ 3 ];
		if( elements != elementsPadding )
		{
			logError( u8"The input data is expected to be dense, i.e. no padding" );
			return E_INVALIDARG;
		}

		const size_t bufferBytes = elements * bytesPerElement( desc.dataType );
		if( bufferBytes > length )
		{
			logError( u8"Tensor length does not match" );
			return E_FAIL;
		}

		if( !weightCompressor )
			CHECK( iCompressor::create( weightCompressor, device ) );

		std::vector<__m256i> buffer;
		CHECK( weightCompressor->getBuffer( buffer, length ) );

		CHECK( stream->read( buffer.data(), length ) );

		CHECK( weightCompressor->bcml( pp, desc, buffer, bufferBytes ) );

		return S_OK;
	}
	else
	{
		// The input is a tensor which was already compressed
		CComPtr<iTensor> res;
		CHECK( createUninitializedTensor( &res, desc ) );
		CHECK( loadTensor( res, stream, length ) );
		*pp = res.Detach();
		return S_OK;
	}
}

HRESULT Device::loadSentencePieceModel( SentencePiece::iProcessor** pp, ComLight::iReadStream* stream, uint32_t length ) noexcept
{
	return SentencePiece::loadSentencePieceModel( pp, stream, length );
}

HRESULT Device::createUninitializedTensor( iTensor** pp, const sTensorDesc& desc ) noexcept
{
	if( nullptr == pp )
		return E_POINTER;

	if( desc.usage == eBufferUse::Immutable )
	{
		ComLight::CComPtr<ComLight::Object<Tensor>> result;
		CHECK( ComLight::Object<Tensor>::create( result, desc, nullptr ) );
		result.detach( pp );
		return S_OK;
	}

	return E_NOTIMPL;
}

HRESULT Device::loadTensor( iTensor* tensor, ComLight::iReadStream* stream, uint32_t length ) noexcept
{
	if( nullptr == tensor || nullptr == stream )
		return E_POINTER;

	Tensor* tensorBase = static_cast<Tensor*>( tensor );
	// Ensure it's empty
	if( nullptr != tensorBase->readView() )
	{
		logError( u8"The tensor supplied to iDevice.loadTensor has been already initialized" );
		return E_INVALIDARG;
	}

	LargeBuffer buffer;
	CHECK( buffer.allocate( length ) );
	CHECK( stream->read( buffer.pointer(), length ) );
	return tensorBase->loadData( device, buffer.pointer(), length );
}