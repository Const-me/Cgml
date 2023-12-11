#pragma once
#include "../../ComLightLib/comLightCommon.h"
#include "../../ComLightLib/streams.h"
#include "sDeviceParams.h"
#include "sDeviceInfo.h"
#include "iContext.cl.h"
#include "iSentencePiece.cl.h"

namespace Cgml
{
	struct DECLSPEC_NOVTABLE iDevice : public ComLight::IUnknown
	{
		DEFINE_INTERFACE_ID( "a90d9b4f-be31-495c-abbb-a7b1d1d58c57" );

		virtual HRESULT createTensor( iTensor** pp, const sTensorDesc& desc, iTensor* reuse ) = 0;

		virtual HRESULT uploadImmutableTensor( iTensor** pp, const sTensorDesc& desc, const void* rsi, uint32_t length ) = 0;

		virtual HRESULT createUninitializedTensor( iTensor** pp, const sTensorDesc& desc ) = 0;

		virtual HRESULT loadTensor( iTensor* tensor, ComLight::iReadStream* stream, uint32_t length ) = 0;

		virtual HRESULT loadImmutableTensor( iTensor** pp, const sTensorDesc& desc, ComLight::iReadStream* stream, uint32_t length, eLoadTransform tform ) = 0;

		virtual HRESULT waitForWeightsCompressor() = 0;

		virtual HRESULT getDeviceInfo( sDeviceInfo& rdi ) = 0;

		virtual HRESULT loadSentencePieceModel( SentencePiece::iProcessor** pp, ComLight::iReadStream* stream, uint32_t length ) = 0;
	};

	HRESULT COMLIGHTCALL listGPUs( pfnListAdapters pfn, void* pv );

	HRESULT COMLIGHTCALL createDeviceAndContext( const sDeviceParams& deviceParams, iDevice** device, iContext** context );
}