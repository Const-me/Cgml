#include "stdafx.h"
#include "renderDoc.h"
#include "renderdoc_app.h"
#include "../device.h"

#define ENABLE_RENDERDOC_DEBUGGER 1

#if ENABLE_RENDERDOC_DEBUGGER
namespace
{
	static HMODULE hmRenderDoc = nullptr;
	static RENDERDOC_API_1_6_0* api = nullptr;
}

bool DirectCompute::initializeRenderDoc()
{
	hmRenderDoc = GetModuleHandleW( L"renderdoc.dll" );
	if( nullptr == hmRenderDoc )
		return false;

	pRENDERDOC_GetAPI getApi = (pRENDERDOC_GetAPI)GetProcAddress( hmRenderDoc, "RENDERDOC_GetAPI" );
	if( nullptr == getApi )
		return false;
	if( 1 != getApi( eRENDERDOC_API_Version_1_6_0, (void**)&api ) )
		return false;
	if( nullptr == api )
		return false;

	return true;
}

namespace
{
	using namespace DirectCompute;
	inline bool isKeyPressed( int vKey )
	{
		return 0 != ( GetAsyncKeyState( vKey ) & 0x8000 );
	}
}

CaptureRaii::CaptureRaii( ID3D11Device* dev )
{
	if( nullptr == api )
		return;
	if( !isKeyPressed( VK_F12 ) )
		return;
	if( nullptr == dev )
		return;

	api->StartFrameCapture( dev, nullptr );
	device = dev;
}

CaptureRaii::~CaptureRaii()
{
	if( nullptr == device )
		return;
	api->EndFrameCapture( device, nullptr );
	device = nullptr;
}
#else	// !ENABLE_RENDERDOC_DEBUGGER
bool DirectCompute::initializeRenderDoc()
{
	return false;
}
DirectCompute::CaptureRaii::CaptureRaii( ID3D11Device* dev )
{
}
DirectCompute::CaptureRaii::~CaptureRaii()
{
}
#endif