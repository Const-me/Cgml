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
		virtual HRESULT COMLIGHTCALL bindTensors0( iTensor* result ) = 0;
		virtual HRESULT COMLIGHTCALL bindTensors1( iTensor* result, iTensor* arg0 ) = 0;
		virtual HRESULT COMLIGHTCALL bindTensors2( iTensor* result, iTensor* arg0, iTensor* arg1 ) = 0;
		virtual HRESULT COMLIGHTCALL bindTensors3( iTensor* result, iTensor* arg0, iTensor* arg1, iTensor* arg2 ) = 0;
		virtual HRESULT COMLIGHTCALL bindTensors2w( iTensor* result0, iTensor* result1 ) = 0;
		virtual HRESULT COMLIGHTCALL bindTensors2w2r( iTensor* res0, iTensor* res1, iTensor* source0, iTensor* source1 ) = 0;
		virtual HRESULT COMLIGHTCALL bindTensors2w1r( iTensor* res0, iTensor* res1, iTensor* arg0 ) = 0;
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