#include "stdafx.h"
#include "ConstantBuffersPool.h"
#include "../../ComLightLib/hresult.h"
#include <ammintrin.h>

ConstantBuffersPool::ConstantBuffersPool( size_t maxVectors )
{
	pool.resize( maxVectors + 1 );
}

static HRESULT __declspec( noinline ) createConstantBuffer( ID3D11Device* dev, CComPtr<ID3D11Buffer>& rdi, const uint8_t* constantBufferData, const uint32_t cbSize )
{
	const uint32_t bufferSize = _andn_u32( 15, cbSize + 15 );

	CD3D11_BUFFER_DESC desc{ bufferSize, D3D11_BIND_CONSTANT_BUFFER, D3D11_USAGE_DYNAMIC, D3D11_CPU_ACCESS_WRITE };
	uint8_t* const buffer = (uint8_t*)_alloca( (size_t)bufferSize );
	__movsb( buffer, constantBufferData, cbSize );
	if( cbSize < bufferSize )
		__stosb( buffer + cbSize, 0, bufferSize - cbSize );

	D3D11_SUBRESOURCE_DATA srd{ buffer, 0, 0 };
	CHECK( dev->CreateBuffer( &desc, &srd, &rdi ) );
	return S_OK;
}

HRESULT ConstantBuffersPool::updateAndBind( ID3D11Device* dev, ID3D11DeviceContext* ctx, const uint8_t* constantBufferData, int cbSize )
{
	const int cbSizeVectors = ( cbSize + 15 ) / 16;
	if( cbSizeVectors < 0 || cbSizeVectors >= pool.size() )
		return E_BOUNDS;

	ID3D11Buffer* buffer = pool[ cbSizeVectors ];

	if( nullptr != buffer )
	{
		D3D11_MAPPED_SUBRESOURCE mapped;
		CHECK( ctx->Map( buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped ) );
		__movsb( (uint8_t*)mapped.pData, constantBufferData, cbSize );
		ctx->Unmap( buffer, 0 );
	}
	else if( cbSizeVectors != 0 )
	{
		CHECK( createConstantBuffer( dev, pool[ cbSizeVectors ], constantBufferData, (uint32_t)cbSize ) );
		buffer = pool[ cbSizeVectors ];
	}

	ctx->CSSetConstantBuffers( 0, 1, &buffer );
	return S_OK;
}