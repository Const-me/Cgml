#pragma once
#include "../API/iDevice.cl.h"
#include "../../ComLightLib/comLightServer.h"
#include "../Utils/Compression/iCompressor.h"

namespace Cgml
{
	class Device : public ComLight::ObjectRoot<iDevice>
	{
		HRESULT createTensor( iTensor** pp, const sTensorDesc& desc, iTensor* reuse ) noexcept override final;

		HRESULT loadImmutableTensor( iTensor** pp, const sTensorDesc& desc, ComLight::iReadStream* stream, uint32_t length, eLoadTransform tform ) noexcept override final;
		HRESULT uploadImmutableTensor( iTensor** pp, const sTensorDesc& desc, const void* rsi, uint32_t length ) noexcept override final;
		HRESULT uploadImmutable( iTensor** pp, const sTensorDesc& desc, const void* rsi, size_t length, DXGI_FORMAT format, UINT bufferElements );

		HRESULT getDeviceInfo( sDeviceInfo& rdi ) noexcept override final;

		HRESULT waitForWeightsCompressor() noexcept override final;

		HRESULT loadSentencePieceModel( SentencePiece::iProcessor** pp, ComLight::iReadStream* stream, uint32_t length ) noexcept override final;

		HRESULT createUninitializedTensor( iTensor** pp, const sTensorDesc& desc ) noexcept override final;

		HRESULT loadTensor( iTensor* tensor, ComLight::iReadStream* stream, uint32_t length ) noexcept override final;

		CComPtr<ID3D11Device> device;
		std::wstring deviceName;
		std::unique_ptr<iCompressor> weightCompressor;
		HRESULT loadCompressed( iTensor** pp, const sTensorDesc& desc, ComLight::iReadStream* stream, uint32_t length ) noexcept;

	public:

		Device( ID3D11Device* dev ) :
			device( dev )
		{ }
	};
}