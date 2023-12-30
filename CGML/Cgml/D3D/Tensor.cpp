#include "stdafx.h"
#include "Tensor.h"
#include "tensorUtils.h"
#include "createBuffer.h"
#include "tensorInterop.h"
using namespace Cgml;

HRESULT Tensor::createImmutableRaw( ID3D11Device* device, const std::vector<uint32_t>& data )
{
	if( !srv )
		return createImmutableRawBuffer( device, data, srv );
	return HRESULT_FROM_WIN32( ERROR_ALREADY_INITIALIZED );
}

HRESULT Tensor::loadData( ID3D11Device* device, const uint8_t* rsi, size_t cb ) noexcept
{
	assert( nullptr == srv );

	if( desc.layout == eTensorLayout::Dense )
	{
		sTypeInfo ti;
		CHECK( sTypeInfo::initialize( ti, desc ) );

		size_t elts = desc.shape.countElements();
		const size_t expectedBytes = elts * ti.cbElement;
		if( expectedBytes != cb )
			return E_INVALIDARG;

		return createImmutableBuffer( device, rsi, cb, elts, ti.format, srv );
	}
	else
	{
		const size_t expectedBytes = (size_t)desc.shape.stride[ 3 ] * desc.shape.size[ 3 ];
		if( expectedBytes != cb )
			return E_INVALIDARG;

		return createImmutableRawBuffer( device, (const uint32_t*)rsi, cb / 4, srv );
	}
}

HRESULT ResizeableTensor::resize( ID3D11Device* dev, const sTensorDesc& desc ) noexcept
{
	if( desc.usage != this->desc.usage )
	{
		logError( u8"Can’t change usage with resizing" );
		return E_INVALIDARG;
	}
	if( desc.dataType != this->desc.dataType )
	{
		logError( u8"Can’t change data type with resizing" );
		return E_INVALIDARG;
	}
	if( desc.layout != eTensorLayout::Dense )
	{
		logError( u8"Resizeable tensors don’t support compression" );
		return E_NOTIMPL;
	}

	size_t cap = desc.shape.countElements();
	if( cap > capacity )
	{
		BufferDesc bufferDesc;
		CHECK( bufferDesc.create( desc, capacity ) );
		CHECK( resizeImpl( dev, bufferDesc ) );;
		capacity = bufferDesc.length;
	}

	this->desc = desc;
	return S_OK;
}

HRESULT ResizeableTensor::resizeImpl( ID3D11Device* dev, const BufferDesc& desc ) noexcept
{
	return createDefault( dev, desc, srv, uav );
}

HRESULT ResizeableTensor::loadData( ID3D11Device* device, const uint8_t* rsi, size_t cb ) noexcept
{
	logError( u8"So far, iDevice.loadTensor() method only supports immutable tensors" );
	return E_NOTIMPL;
}

HRESULT DynamicTensor::resizeImpl( ID3D11Device* dev, const BufferDesc& desc ) noexcept
{
	return createDynamic( dev, desc, srv, buffer );
}

HRESULT StagingTensor::resizeImpl( ID3D11Device* dev, const BufferDesc& desc ) noexcept
{
	return createStaging( dev, desc, srv, uav, bufferVram, bufferStaging );
}

HRESULT ResizeableTensor::replaceData( ID3D11DeviceContext* context, pfnUpdateDynamicTensor pfn, void* pv, uint32_t bytes )
{
	return E_NOTIMPL;
}

HRESULT DynamicTensor::replaceData( ID3D11DeviceContext* context, pfnUpdateDynamicTensor pfn, void* pv, uint32_t bytes )
{
	D3D11_MAPPED_SUBRESOURCE mapped;
	CHECK( context->Map( buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped ) );
	const HRESULT hrCallback = pfn( mapped.pData, bytes, pv );
	context->Unmap( buffer, 0 );
	CHECK( hrCallback );
	return S_OK;
}

HRESULT ResizeableTensor::reshape( const TensorShape& shape )
{
	if( capacity >= shape.countElements() )
	{
		desc.shape = shape;
		return S_OK;
	}
	logError( u8"ResizeableTensor.reshape(): capacity is not enough" );
	return E_UNEXPECTED;
}

HRESULT COMLIGHTCALL Tensor::view( const TensorShape& newShape ) noexcept
{
	if( desc.layout != eTensorLayout::Dense )
	{
		logError( u8"iTensor.view() requires a dense tensor" );
		return E_INVALIDARG;
	}

	size_t currentElts = desc.shape.countElements();
	size_t newElements = newShape.countElements();
	if( currentElts != newElements )
	{
		logError( u8"iTensor.view() can’t be user to resize tensors, but asked to resize %zu -> %zu elements",
			currentElts, newElements );
		return E_INVALIDARG;
	}

	desc.shape = newShape;
	return S_OK;
}

HRESULT COMLIGHTCALL Tensor::getMemoryUse( __m128i* rdi ) const noexcept
{
	__m128i buffer = resourceMemoryUsage( srv );
	__m128i obj = setLow_size( sizeof( ComLight::Object<Tensor> ) );
	__m128i cb = _mm_add_epi64( buffer, obj );
	_mm_storeu_si128( rdi, cb );
	return S_OK;
}

HRESULT COMLIGHTCALL DynamicTensor::getMemoryUse( __m128i* rdi ) const noexcept
{
	__m128i buffer = resourceMemoryUsage( srv );
	__m128i obj = setLow_size( sizeof( ComLight::Object<DynamicTensor> ) );
	__m128i cb = _mm_add_epi64( buffer, obj );
	_mm_storeu_si128( rdi, cb );
	return S_OK;
}

HRESULT COMLIGHTCALL StagingTensor::getMemoryUse( __m128i* rdi ) const noexcept
{
	__m128i v = resourceMemoryUsage( srv );
	v = _mm_add_epi64( v, bufferMemoryUsage( bufferStaging ) );
	v = _mm_add_epi64( v, setLow_size( sizeof( ComLight::Object<StagingTensor> ) ) );
	_mm_storeu_si128( rdi, v );
	return S_OK;
}

HRESULT StagingTensor::replaceData( ID3D11DeviceContext* context, pfnUpdateDynamicTensor pfn, void* pv, uint32_t bytes )
{
	D3D11_MAPPED_SUBRESOURCE mapped;
	CHECK( context->Map( bufferStaging, 0, D3D11_MAP_WRITE, 0, &mapped ) );
	const HRESULT hrCallback = pfn( mapped.pData, bytes, pv );
	context->Unmap( bufferStaging, 0 );
	CHECK( hrCallback );

	context->CopyResource( bufferVram, bufferStaging );
	return S_OK;
}

static HRESULT downloadImpl( ID3D11DeviceContext* context, pfnReadTensor pfn, void* pv, ID3D11Buffer* bufferStaging, uint32_t bytes )
{
	D3D11_MAPPED_SUBRESOURCE mapped;
	CHECK( context->Map( bufferStaging, 0, D3D11_MAP_READ, 0, &mapped ) );

	HRESULT hrCallback = pfn( mapped.pData, bytes, pv );
	context->Unmap( bufferStaging, 0 );
	CHECK( hrCallback );

	return S_OK;
}

// Make D3D11_BOX to extract initial portion of a buffer
static inline D3D11_BOX bufferSourceBox( uint32_t cb )
{
	D3D11_BOX box;
	__m128i v = _mm_setzero_si128();
	v = _mm_insert_epi32( v, (int)cb, 3 );
	_mm_storeu_si128( ( __m128i* )( &box ), v );
	*(uint64_t*)&box.bottom = 0x100000001ull;
	return box;
}

HRESULT Tensor::download( ID3D11DeviceContext* context, pfnReadTensor pfn, void* pv, eDownloadFlag flag, bool skipWarning ) const noexcept
{
	if( nullptr == pfn )
		return E_POINTER;
	if( nullptr == srv )
		return OLE_E_BLANK;

	if( desc.layout != eTensorLayout::Dense )
	{
		logError( u8"Downloading data requires dense memory layout of the tensor" );
		return E_INVALIDARG;
	}

	if( flag != eDownloadFlag::None )
	{
		logError( u8"Flags require eBufferUse.ReadWriteDownload usage" );
		return E_INVALIDARG;
	}

#ifdef NDEBUG
	if( !skipWarning )
		logWarning( u8"For optimal performance, iContext.download should only be used with eBufferUse.ReadWriteDownload tensors" );
#endif

	CComPtr<ID3D11Resource> resource;
	srv->GetResource( &resource );
	CComPtr<ID3D11Buffer> sourceBuffer;
	CHECK( resource.QueryInterface( &sourceBuffer ) );

	const size_t bytes = desc.shape.countElements() * bytesPerElement( desc.dataType );
	if( bytes > INT_MAX )
		return DISP_E_OVERFLOW;

	CComPtr<ID3D11Device> dev;
	context->GetDevice( &dev );

	CD3D11_BUFFER_DESC desc{ (uint32_t)bytes, 0, D3D11_USAGE_STAGING, D3D11_CPU_ACCESS_READ };
	CComPtr<ID3D11Buffer> bufferStaging;
	CHECK( dev->CreateBuffer( &desc, nullptr, &bufferStaging ) );

	const D3D11_BOX box = bufferSourceBox( (uint32_t)bytes );
	context->CopySubresourceRegion( bufferStaging, 0, 0, 0, 0, sourceBuffer, 0, &box );

	return downloadImpl( context, pfn, pv, bufferStaging, (uint32_t)bytes );
}

HRESULT StagingTensor::download( ID3D11DeviceContext* context, pfnReadTensor pfn, void* pv, eDownloadFlag flag, bool skipWarning ) const noexcept
{
	if( desc.layout != eTensorLayout::Dense )
	{
		logError( u8"Downloading data requires dense memory layout of the tensor" );
		return E_INVALIDARG;
	}

	const size_t bytes = desc.shape.countElements() * bytesPerElement( desc.dataType );
	if( bytes > INT_MAX )
		return DISP_E_OVERFLOW;

	if( flag != eDownloadFlag::ReadStaging )
	{
		const D3D11_BOX box = bufferSourceBox( (uint32_t)bytes );
		context->CopySubresourceRegion( bufferStaging, 0, 0, 0, 0, bufferVram, 0, &box );
	}

	if( flag != eDownloadFlag::CopyToStaging )
		return downloadImpl( context, pfn, pv, bufferStaging, (uint32_t)bytes );

	return S_OK;
}

ID3D11UnorderedAccessView* Cgml::getTensorUav( iTensor* tensor )
{
	if( nullptr == tensor )
		return nullptr;

	Tensor* t = static_cast<Tensor*>( tensor );
	return t->writeView();
}