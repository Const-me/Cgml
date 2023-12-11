#include "stdafx.h"
#include "Context.h"
#include "Tensor.h"
#include "tensorUtils.h"
using namespace Cgml;

HRESULT COMLIGHTCALL Context::writeDynamic( iTensor* tensor, const TensorShape& shape, pfnUpdateDynamicTensor pfn, void* pv ) noexcept
{
	if( nullptr == tensor || nullptr == pfn )
		return E_POINTER;

	Tensor* tensorBase = static_cast<Tensor*>( tensor );
	const eBufferUse usage = tensorBase->getDesc().usage;
	if( usage == eBufferUse::Immutable )
	{
		logError( u8"Immutable tensors don’t support updates" );
		return E_INVALIDARG;
	}

	ResizeableTensor* rt = static_cast<ResizeableTensor*>( tensorBase );
	sTensorDesc desc = rt->getDesc();
	desc.shape = shape;
	CHECK( rt->resize( device, desc ) );

	const uint32_t mappedBytes = (uint32_t)( shape.countElements() * bytesPerElement( desc.dataType ) );
	CHECK( rt->replaceData( context, pfn, pv, mappedBytes ) );
	return S_OK;
}

HRESULT COMLIGHTCALL Context::download( iTensor* tensor, pfnReadTensor pfn, void* pv, eDownloadFlag flag ) noexcept
{
	if( nullptr == tensor )
		return E_POINTER;
	if( nullptr == pfn && flag != eDownloadFlag::CopyToStaging )
		return E_POINTER;

	const Tensor* tensorBase = static_cast<Tensor*>( tensor );
	return tensorBase->download( context, pfn, pv, flag );
}

static HRESULT __stdcall writeDataCallback( const void* rsi, uint32_t cb, void* pv )
{
	ComLight::iWriteStream* const stream = (ComLight::iWriteStream*)pv;
	return stream->write( rsi, (int)cb );
}

HRESULT COMLIGHTCALL Context::writeTensorData( iTensor* tensor, ComLight::iWriteStream* stream ) noexcept
{
	if( nullptr == tensor || nullptr == stream )
		return E_POINTER;

	const Tensor* tensorBase = static_cast<Tensor*>( tensor );

	// return tensorBase->download( context, &writeDataCallback, stream, true );

	if( tensorBase->getDesc().usage != eBufferUse::Immutable )
	{
		logError( u8"So far, saveTensorData only supports immutable tensors" );
		return E_NOTIMPL;
	}

	ID3D11ShaderResourceView* const srv = tensorBase->readView();
	if( nullptr == srv )
		return OLE_E_BLANK;

	CComPtr<ID3D11Resource> resource;
	srv->GetResource( &resource );

	CComPtr<ID3D11Buffer> buffer;
	CHECK( resource.QueryInterface( &buffer ) );

	D3D11_BUFFER_DESC desc;
	buffer->GetDesc( &desc );

	desc.Usage = D3D11_USAGE_STAGING;
	desc.BindFlags = 0;
	desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
	desc.MiscFlags = 0;

	CComPtr<ID3D11Buffer> staging;
	CHECK( device->CreateBuffer( &desc, nullptr, &staging ) );
	context->CopyResource( staging, buffer );

	D3D11_MAPPED_SUBRESOURCE mapped;
	CHECK( context->Map( staging, 0, D3D11_MAP_READ, 0, &mapped ) );
	const HRESULT hr = stream->write( mapped.pData, (int)desc.ByteWidth );
	context->Unmap( staging, 0 );

	return hr;
}

HRESULT COMLIGHTCALL Context::copy( iTensor* destination, iTensor* source ) noexcept
{
	if( nullptr == destination || nullptr == source )
		return E_POINTER;
	Tensor* destBase = static_cast<Tensor*>( destination );
	if( destBase->getDesc().usage == eBufferUse::Immutable )
	{
		logError( u8"iContext.copy asked to write into an immutable tensor" );
		return E_INVALIDARG;
	}
	Tensor* sourceBase = static_cast<Tensor*>( source );

	// TODO: moar argument validations
	// Fail with compressed tensors, require identical data types, require equal size..
	ID3D11ShaderResourceView* srvDest = destBase->readView();
	ID3D11ShaderResourceView* srvSource = sourceBase->readView();
	if( nullptr == srvDest || nullptr == srvSource )
		return OLE_E_BLANK;

	CComPtr<ID3D11Resource> dest, src;
	srvDest->GetResource( &dest );
	srvSource->GetResource( &src );

	context->CopyResource( dest, src );
	return S_OK;
}