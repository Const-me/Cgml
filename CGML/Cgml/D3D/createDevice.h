#pragma once
#include "../API/sDeviceParams.h"

namespace DirectCompute
{
	struct ComputeDevice
	{
		CComPtr<ID3D11Device> device;
		CComPtr<ID3D11DeviceContext> context;
	};

	HRESULT create( const Cgml::sDeviceParams& deviceParams, ComputeDevice& rdi );
}