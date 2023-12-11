#pragma once
#include <API/sTensorDesc.h>

namespace Cgml
{
	HRESULT loadTransform( eLoadTransform tform, eDataType& dt, DXGI_FORMAT& viewFormat, void* pv, size_t elements );
}