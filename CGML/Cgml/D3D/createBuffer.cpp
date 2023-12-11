#include "stdafx.h"
#include "createBuffer.h"
#include "tensorUtils.h"

namespace Cgml
{
	HRESULT BufferDesc::create( const sTensorDesc& desc )
	{
		sTypeInfo ti;
		CHECK( sTypeInfo::initialize( ti, desc ) );

		usage = desc.usage;
		cbElement = ti.cbElement;
		dxgiFormat = ti.format;
		isRaw = ti.isRaw;

		const size_t length = horizontalProduct( desc.shape.sizeVec() );
		const size_t width = length * ti.cbElement;
		if( 0 != width >> 31 )
		{
			logError( u8"Buffer size exceeds 2GB" );
			return DISP_E_OVERFLOW;
		}

		this->length = (uint32_t)length;
		return S_OK;
	}

	HRESULT BufferDesc::create( const sTensorDesc& desc, uint32_t oldCapacity )
	{
		CHECK( create( desc ) );
		// TODO: adjust size to reduce count of buffer re-allocations
		return S_OK;
	}

	static HRESULT createBuffer( ID3D11Device* device, const BufferDesc& desc,
		CComPtr<ID3D11ShaderResourceView>& srv, ID3D11UnorderedAccessView** uav,
		ID3D11Buffer** bufferVram, ID3D11Buffer** bufferStaging )
	{
		srv = nullptr;

		if( desc.isRaw )
		{
			logError( u8"Byte address buffers aren’t implemented yet" );
			return E_NOTIMPL;
		}
		const size_t bufferBytes = desc.byteWidth();

		CComPtr<ID3D11Buffer> gpuBuffer;
		CComPtr<ID3D11Buffer> stagingBuffer;
		{
			CD3D11_BUFFER_DESC bufferDesc{ (uint32_t)bufferBytes, D3D11_BIND_SHADER_RESOURCE };
			switch( desc.usage )
			{
			case eBufferUse::ReadWrite:
			case eBufferUse::ReadWriteDownload:
				bufferDesc.BindFlags |= D3D11_BIND_UNORDERED_ACCESS;
				break;
			case eBufferUse::Dynamic:
				bufferDesc.Usage = D3D11_USAGE_DYNAMIC;
				bufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
				break;
			default:
				return E_UNEXPECTED;
			}

			HRESULT hr = device->CreateBuffer( &bufferDesc, nullptr, &gpuBuffer );
			if( FAILED( hr ) )
			{
				logErrorHr( hr, u8"ID3D11Device.CreateBuffer failed" );
				return hr;
			}

			if( desc.usage == eBufferUse::ReadWriteDownload )
			{
				bufferDesc.BindFlags = 0;
				bufferDesc.Usage = D3D11_USAGE_STAGING;
				bufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ | D3D11_CPU_ACCESS_WRITE;

				hr = device->CreateBuffer( &bufferDesc, nullptr, &stagingBuffer );
				if( FAILED( hr ) )
				{
					logErrorHr( hr, u8"ID3D11Device.CreateBuffer failed" );
					return hr;
				}
			}
		}

		{
			CD3D11_SHADER_RESOURCE_VIEW_DESC viewDesc{ D3D11_SRV_DIMENSION_BUFFER, desc.format(), 0, desc.length };
			HRESULT hr = device->CreateShaderResourceView( gpuBuffer, &viewDesc, &srv );
			if( FAILED( hr ) )
			{
				logErrorHr( hr, u8"ID3D11Device.CreateShaderResourceView failed" );
				return hr;
			}
		}

		if( desc.usage == eBufferUse::Dynamic )
		{
			if( nullptr == bufferVram )
				return E_POINTER;
			*bufferVram = gpuBuffer.Detach();
			return S_OK;
		}

		if( nullptr == uav )
			return E_POINTER;

		{
			CD3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc{ D3D11_UAV_DIMENSION_BUFFER, desc.format(), 0, desc.length };
			HRESULT hr = device->CreateUnorderedAccessView( gpuBuffer, &uavDesc, uav );
			if( FAILED( hr ) )
			{
				logErrorHr( hr, u8"ID3D11Device.CreateUnorderedAccessView failed" );
				return hr;
			}
		}

		if( desc.usage == eBufferUse::ReadWrite )
			return S_OK;
		if( desc.usage != eBufferUse::ReadWriteDownload )
			return E_UNEXPECTED;

		if( nullptr == bufferVram || nullptr == bufferStaging )
			return E_POINTER;
		*bufferVram = gpuBuffer.Detach();
		*bufferStaging = stagingBuffer.Detach();
		return S_OK;
	}

	HRESULT createDefault( ID3D11Device* dev, const BufferDesc& desc,
		CComPtr<ID3D11ShaderResourceView>& srv, CComPtr<ID3D11UnorderedAccessView>& uav )
	{
		if( desc.usage != eBufferUse::ReadWrite )
			return E_INVALIDARG;

		uav = nullptr;
		return createBuffer( dev, desc, srv, &uav, nullptr, nullptr );
	}

	HRESULT createDynamic( ID3D11Device* dev, const BufferDesc& desc,
		CComPtr<ID3D11ShaderResourceView>& srv, CComPtr<ID3D11Buffer>& buffer )
	{
		if( desc.usage != eBufferUse::Dynamic )
			return E_INVALIDARG;

		buffer = nullptr;
		return createBuffer( dev, desc, srv, nullptr, &buffer, nullptr );
	}

	HRESULT createStaging( ID3D11Device* dev, const BufferDesc& desc,
		CComPtr<ID3D11ShaderResourceView>& srv, CComPtr<ID3D11UnorderedAccessView>& uav,
		CComPtr<ID3D11Buffer>& bufferVram, CComPtr<ID3D11Buffer>& bufferStaging )
	{
		if( desc.usage != eBufferUse::ReadWriteDownload )
			return E_INVALIDARG;

		uav = nullptr;
		bufferVram = nullptr;
		bufferStaging = nullptr;
		return createBuffer( dev, desc, srv, &uav, &bufferVram, &bufferStaging );
	}

	HRESULT createImmutableRawBuffer( ID3D11Device* dev, const uint32_t* rsi, size_t length, CComPtr<ID3D11ShaderResourceView>& srv )
	{
		srv = nullptr;
		const size_t byteWidth = length * 4;
		if( byteWidth > INT_MAX )
			return DISP_E_OVERFLOW;

		CComPtr<ID3D11Buffer> gpuBuffer;
		{
			CD3D11_BUFFER_DESC bufferDesc{ (uint32_t)byteWidth, D3D11_BIND_SHADER_RESOURCE, D3D11_USAGE_IMMUTABLE, 0, D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS };
			D3D11_SUBRESOURCE_DATA srd{ rsi, 0, 0 };
			CHECK( dev->CreateBuffer( &bufferDesc, &srd, &gpuBuffer ) );
		}

		{
			D3D11_SHADER_RESOURCE_VIEW_DESC desc;
			memset( &desc, 0, sizeof( desc ) );
			desc.Format = DXGI_FORMAT_R32_TYPELESS;
			desc.ViewDimension = D3D_SRV_DIMENSION_BUFFEREX;
			desc.BufferEx.NumElements = (uint32_t)length;
			desc.BufferEx.Flags = D3D11_BUFFEREX_SRV_FLAG_RAW;

			CHECK( dev->CreateShaderResourceView( gpuBuffer, &desc, &srv ) );
		}

		return S_OK;
	}

	HRESULT createImmutableBuffer( ID3D11Device* device, const void* rsi, size_t byteWidth,
		size_t viewSize, DXGI_FORMAT viewFormat, CComPtr<ID3D11ShaderResourceView>& srv )
	{
		srv = nullptr;

		CComPtr<ID3D11Buffer> gpuBuffer;
		{
			CD3D11_BUFFER_DESC desc{ (uint32_t)byteWidth, D3D11_BIND_SHADER_RESOURCE, D3D11_USAGE_IMMUTABLE };
			D3D11_SUBRESOURCE_DATA srd{ rsi, 0, 0 };
			HRESULT hr = device->CreateBuffer( &desc, &srd, &gpuBuffer );
			if( FAILED( hr ) )
			{
				logErrorHr( hr, u8"ID3D11Device.CreateBuffer failed" );
				return hr;
			}
		}

		{
			CD3D11_SHADER_RESOURCE_VIEW_DESC desc{ D3D11_SRV_DIMENSION_BUFFER, viewFormat, 0, (uint32_t)viewSize };
			HRESULT hr = device->CreateShaderResourceView( gpuBuffer, &desc, &srv );
			if( FAILED( hr ) )
			{
				logErrorHr( hr, u8"ID3D11Device.CreateShaderResourceView failed" );
				return hr;
			}
		}

		return S_OK;
	}
}