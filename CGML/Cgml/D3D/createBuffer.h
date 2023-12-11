#pragma once
#include "../API/sTensorDesc.h"

namespace Cgml
{
	struct BufferDesc
	{
		eBufferUse usage;
		uint8_t cbElement;
		uint8_t dxgiFormat;
		bool isRaw;
		uint32_t length;

		HRESULT create( const sTensorDesc& desc );
		HRESULT create( const sTensorDesc& desc, uint32_t oldCapacity );

		size_t byteWidth() const
		{
			return (size_t)length * cbElement;
		}

		DXGI_FORMAT format() const
		{
			return (DXGI_FORMAT)dxgiFormat;
		}
	};

	HRESULT createDefault( ID3D11Device* dev, const BufferDesc& desc,
		CComPtr<ID3D11ShaderResourceView>& srv, CComPtr<ID3D11UnorderedAccessView>& uav );

	HRESULT createDynamic( ID3D11Device* dev, const BufferDesc& desc,
		CComPtr<ID3D11ShaderResourceView>& srv, CComPtr<ID3D11Buffer>& buffer );

	HRESULT createStaging( ID3D11Device* dev, const BufferDesc& desc,
		CComPtr<ID3D11ShaderResourceView>& srv, CComPtr<ID3D11UnorderedAccessView>& uav,
		CComPtr<ID3D11Buffer>& bufferVram, CComPtr<ID3D11Buffer>& bufferStaging );

	HRESULT createImmutableRawBuffer( ID3D11Device* dev, const uint32_t* rsi, size_t length, CComPtr<ID3D11ShaderResourceView>& srv );

	inline HRESULT createImmutableRawBuffer( ID3D11Device* dev, const std::vector<uint32_t>& data, CComPtr<ID3D11ShaderResourceView>& srv )
	{
		if( !data.empty() )
			return createImmutableRawBuffer( dev, data.data(), data.size(), srv );
		else
			return E_INVALIDARG;
	}

	HRESULT createImmutableBuffer( ID3D11Device* dev, const void* rsi, size_t byteWidth,
		size_t viewSize, DXGI_FORMAT viewFormat, CComPtr<ID3D11ShaderResourceView>& srv );
}