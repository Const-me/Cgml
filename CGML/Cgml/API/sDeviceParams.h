#pragma once

namespace Cgml
{
	struct sDeviceParams
	{
		const wchar_t* adapter = nullptr;
		int queueLength;
		uint8_t flags;

		static const uint8_t FLAG_POWERSAVER = 1;
	};

	using pfnListAdapters = void( __stdcall* )( const wchar_t* name, void* pv );
}