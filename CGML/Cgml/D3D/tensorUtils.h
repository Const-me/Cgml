#pragma once
#include <API/sTensorDesc.h>

namespace Cgml
{
	struct sTypeInfo
	{
		DXGI_FORMAT format;
		uint8_t cbElement;
		bool isRaw;

		static HRESULT initialize( sTypeInfo& rdi, const sTensorDesc& desc );
	};

	size_t bytesPerElement( eDataType dt );
}