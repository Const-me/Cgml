#pragma once
#include <API/sTensorDesc.h>

namespace Cgml
{
	// Apply tensor load transformation, in place
	HRESULT loadTransform( eLoadTransform tform, eDataType& dt, DXGI_FORMAT& viewFormat, void* pv, size_t& bytes, size_t elements );
}