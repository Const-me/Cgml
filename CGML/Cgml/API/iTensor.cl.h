#pragma once
#include "../../ComLightLib/comLightCommon.h"
#include "sTensorDesc.h"

namespace Cgml
{
	struct DECLSPEC_NOVTABLE iTensor : public ComLight::IUnknown
	{
		DEFINE_INTERFACE_ID( "30d2adce-78d3-4881-85e2-f7f00e898a2f" );

		virtual HRESULT COMLIGHTCALL getDesc( sTensorDesc& desc ) const = 0;

		// Change tensor shape while retaining the data
		virtual HRESULT COMLIGHTCALL view( const TensorShape& newShape ) = 0;

		virtual HRESULT COMLIGHTCALL getMemoryUse( __m128i* rdi ) const = 0;
	};
}