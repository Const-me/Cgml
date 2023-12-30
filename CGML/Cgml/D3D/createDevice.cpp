#include "stdafx.h"
#include "createDevice.h"
#include "Device.h"
#include "Context.h"
#include "listGPUs.h"
#include <ammintrin.h>
#pragma comment(lib, "D3D11.lib")
#include "RenderDoc/renderDoc.h"

namespace DirectCompute
{
	HRESULT create( const Cgml::sDeviceParams& deviceParams, ComputeDevice& rdi )
	{
		CComPtr<IDXGIAdapter1> adapter = DirectCompute::selectAdapter( deviceParams.adapter );
		const D3D_DRIVER_TYPE driverType = adapter ? D3D_DRIVER_TYPE_UNKNOWN : D3D_DRIVER_TYPE_HARDWARE;

		const std::array<D3D_FEATURE_LEVEL, 4> levels = { D3D_FEATURE_LEVEL_12_1 , D3D_FEATURE_LEVEL_12_0 , D3D_FEATURE_LEVEL_11_1 , D3D_FEATURE_LEVEL_11_0 };
		UINT flags = D3D11_CREATE_DEVICE_DISABLE_GPU_TIMEOUT | D3D11_CREATE_DEVICE_SINGLETHREADED | D3D11_CREATE_DEVICE_BGRA_SUPPORT;
		bool renderDoc = DirectCompute::initializeRenderDoc();
#ifdef _DEBUG
		if( !renderDoc )
		{
			// Last time I checked, RenderDoc crashed with debug version of D3D11 runtime
			// Only setting this flag unless renderdoc.dll is loaded to the current process
			flags |= D3D11_CREATE_DEVICE_DEBUG;
		}
#endif
		constexpr UINT levelsCount = (UINT)levels.size();
		HRESULT hr = D3D11CreateDevice( adapter, driverType, nullptr, flags, levels.data(), levelsCount, D3D11_SDK_VERSION, &rdi.device, nullptr, &rdi.context );
		if( SUCCEEDED( hr ) )
			return S_OK;

		// D3D11_CREATE_DEVICE_DISABLE_GPU_TIMEOUT: This value is not supported until Direct3D 11.1
// https://learn.microsoft.com/en-us/windows/win32/api/d3d11/ne-d3d11-d3d11_create_device_flag
		flags = _andn_u32( D3D11_CREATE_DEVICE_DISABLE_GPU_TIMEOUT, flags );

		hr = D3D11CreateDevice( adapter, driverType, nullptr, flags, levels.data(), levelsCount, D3D11_SDK_VERSION, &rdi.device, nullptr, &rdi.context );
		if( SUCCEEDED( hr ) )
			return S_OK;
		return hr;
	}
}

HRESULT COMLIGHTCALL Cgml::createDeviceAndContext( const sDeviceParams& deviceParams, iDevice** device, iContext** context )
{
	if( nullptr == device || nullptr == context )
		return E_POINTER;

	using namespace DirectCompute;
	ComputeDevice computeDevice;
	CHECK( create( deviceParams, computeDevice ) );

	ComLight::CComPtr<ComLight::Object<Device>> dev;
	CHECK( ComLight::Object<Device>::create( dev, computeDevice.device ) );

	int queueLength = deviceParams.queueLength;
	if( queueLength < 2 )
	{
		logWarning( u8"sDeviceParams.queueLength is too small, clamped to 2" );
		queueLength = 2;
	}
	else if( queueLength > 64 )
	{
		logWarning( u8"sDeviceParams.queueLength is too large, clamped to 64" );
		queueLength = 64;
	}
	const bool powerSaver = 0 != ( deviceParams.flags & sDeviceParams::FLAG_POWERSAVER );

	ComLight::CComPtr<ComLight::Object<Context>> ctx;
	CHECK( ComLight::Object<Context>::create( ctx, computeDevice.device, computeDevice.context, (uint32_t)queueLength, powerSaver ) );

	dev.detach( device );
	ctx.detach( context );
	return S_OK;
}