#pragma once

namespace DirectCompute
{
	CComPtr<IDXGIAdapter1> selectAdapter( const wchar_t* requestedName );
}