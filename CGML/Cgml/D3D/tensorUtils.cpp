#include "stdafx.h"
#include "tensorUtils.h"

namespace Cgml
{
	size_t bytesPerElement( eDataType dataType )
	{
		switch( dataType )
		{
		case eDataType::FP16:
		case eDataType::BF16:
			return 2;
		case eDataType::FP32:
		case eDataType::U32:
			return 4;
		default:
			logError( u8"Element type %i not yet implemented", (int)dataType );
			return 0;
		}
	}

	HRESULT sTypeInfo::initialize( sTypeInfo& rdi, const sTensorDesc& desc )
	{
		if( desc.layout != eTensorLayout::Dense )
		{
			logError( u8"So far, createTensor method only supports dense tensors" );
			return E_NOTIMPL;
		}
		rdi.isRaw = false;

		switch( desc.dataType )
		{
		case eDataType::FP16:
			rdi.format = DXGI_FORMAT_R16_FLOAT;
			rdi.cbElement = 2;
			break;
		case eDataType::FP32:
			rdi.format = DXGI_FORMAT_R32_FLOAT;
			rdi.cbElement = 4;
			break;
		case eDataType::U32:
			rdi.format = DXGI_FORMAT_R32_UINT;
			rdi.cbElement = 4;
			break;
		case eDataType::BF16:
			rdi.format = DXGI_FORMAT_R16_UINT;
			rdi.cbElement = 2;
			break;
		default:
			logError( u8"Element type %i not yet implemented", (int)desc.dataType );
			return E_NOTIMPL;
		}

		return S_OK;
	}
}