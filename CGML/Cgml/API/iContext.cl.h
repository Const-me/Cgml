#pragma once
#include "iTensor.cl.h"
#include "profiler.h"
#include "../../ComLightLib/streams.h"
#include "eDownloadFlag.h"

namespace Cgml
{
	struct DECLSPEC_NOVTABLE iContext : public ComLight::IUnknown
	{
		DEFINE_INTERFACE_ID( "f249552e-5942-4480-b833-6ba124974f46" );

		virtual HRESULT COMLIGHTCALL bindShader( uint16_t id, const uint8_t* constantBufferData, int cbSize ) = 0;
		virtual HRESULT COMLIGHTCALL dispatch( int groupsX, int groupsY, int groupsZ ) = 0;
		virtual HRESULT COMLIGHTCALL bindTensors( iTensor** arr, int countWrite, int countRead ) = 0;
		virtual HRESULT COMLIGHTCALL unbindInputs() = 0;

		// Copy the entire contents of the source tensor to the destination tensor using the GPU
		virtual HRESULT COMLIGHTCALL copy( iTensor* destination, iTensor* source ) = 0;

		// Replace size, shape and content of the dynamic tensor
		// If the tensor doesn't have the capacity, will be re-created.
		virtual HRESULT COMLIGHTCALL writeDynamic( iTensor* tensor, const TensorShape& shape, pfnUpdateDynamicTensor pfn, void* pv ) = 0;

		// Download tensor data from VRAM to system memory
		virtual HRESULT COMLIGHTCALL download( iTensor* tensor, pfnReadTensor pfn, void* pv, eDownloadFlag flag ) = 0;

		virtual HRESULT COMLIGHTCALL profilerBlockStart( uint16_t id ) = 0;
		virtual HRESULT COMLIGHTCALL profilerBlockEnd() = 0;
		virtual HRESULT COMLIGHTCALL profilerGetData( pfnProfilerData pfn, void* pv ) = 0;

		virtual HRESULT COMLIGHTCALL createComputeShaders( int count, const std::pair<int, int>* blobs, const uint8_t* data, int dataSize ) = 0;

		virtual HRESULT COMLIGHTCALL writeTensorData( iTensor* tensor, ComLight::iWriteStream* stream ) = 0;
	};
}