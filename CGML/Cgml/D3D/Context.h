#pragma once
#include "../API/iContext.cl.h"
#include "../../ComLightLib/comLightServer.h"
#include "ConstantBuffersPool.h"
#include "../Utils/Profiler/GpuProfiler.h"

namespace Cgml
{
	class Context : public ComLight::ObjectRoot<iContext>
	{
		CComPtr<ID3D11Device> device;
		CComPtr<ID3D11DeviceContext> context;

		std::vector<CComPtr<ID3D11ComputeShader>> shaders;
		ConstantBuffersPool constantBuffers;
		uint8_t boundUavs = 0;
		uint8_t boundSrvs = 0;
		GpuProfiler profiler;

		// Copy the entire contents of the source tensor to the destination tensor using the GPU
		HRESULT COMLIGHTCALL copy( iTensor* destination, iTensor* source ) noexcept override final;

		// Replace tensor data with the content of the supplied buffer
		HRESULT COMLIGHTCALL writeDynamic( iTensor* tensor, const TensorShape& shape, pfnUpdateDynamicTensor pfn, void* pv ) noexcept override final;

		// Download tensor data from VRAM to system memory
		HRESULT COMLIGHTCALL download( iTensor* tensor, pfnReadTensor pfn, void* pv, eDownloadFlag flag ) noexcept override final;

		HRESULT COMLIGHTCALL bindShader( uint16_t id, const uint8_t* constantBufferData, int cbSize ) noexcept override final;
		HRESULT COMLIGHTCALL dispatch( int groupsX, int groupsY, int groupsZ ) noexcept override final;

		HRESULT bindTensors( std::initializer_list<iTensor*> uav, std::initializer_list<iTensor*> srv ) noexcept;

		HRESULT COMLIGHTCALL bindTensors0( iTensor* result ) noexcept override final
		{
			return bindTensors( { result }, {} );
		}

		HRESULT COMLIGHTCALL bindTensors1( iTensor* result, iTensor* arg0 ) noexcept override final
		{
			return bindTensors( { result }, { arg0 } );
		}
		HRESULT COMLIGHTCALL bindTensors2( iTensor* result, iTensor* arg0, iTensor* arg1 ) noexcept override final
		{
			return bindTensors( { result }, { arg0, arg1 } );
		}
		HRESULT COMLIGHTCALL bindTensors3( iTensor* result, iTensor* arg0, iTensor* arg1, iTensor* arg2 ) noexcept override final
		{
			return bindTensors( { result }, { arg0, arg1, arg2 } );
		}
		HRESULT COMLIGHTCALL bindTensors2w( iTensor* result0, iTensor* result1 ) noexcept override final
		{
			return bindTensors( { result0, result1 }, {} );
		}
		HRESULT COMLIGHTCALL bindTensors2w2r( iTensor* res0, iTensor* res1, iTensor* source0, iTensor* source1 ) noexcept override final
		{
			return bindTensors( { res0, res1 }, { source0, source1 } );
		}
		HRESULT COMLIGHTCALL bindTensors2w1r( iTensor* res0, iTensor* res1, iTensor* arg0 ) noexcept override final
		{
			return bindTensors( { res0, res1 }, { arg0 } );
		}
		HRESULT COMLIGHTCALL unbindInputs() noexcept override final;

		HRESULT COMLIGHTCALL createComputeShaders( int count, const std::pair<int, int>* blobs, const uint8_t* data, int dataSize ) noexcept override final;

		// HRESULT COMLIGHTCALL sampleTopP( iTensor* dest, iTensor* source, float p ) noexcept override final;

		HRESULT COMLIGHTCALL profilerBlockStart( uint16_t id ) noexcept override final
		{
			return profiler.blockStart( id );
		}

		HRESULT COMLIGHTCALL profilerBlockEnd() noexcept override final
		{
			return profiler.blockEnd();
		}

		HRESULT COMLIGHTCALL profilerGetData( pfnProfilerData pfn, void* pv ) noexcept override final
		{
			return profiler.getData( pfn, pv );
		}

		HRESULT COMLIGHTCALL writeTensorData( iTensor* tensor, ComLight::iWriteStream* stream ) noexcept override final;

	public:
		Context( ID3D11Device* dev, ID3D11DeviceContext* ctx, size_t queueLength, bool powerSaver ) :
			device( dev ),
			context( ctx ),
			profiler( dev, ctx, queueLength, powerSaver )
		{ }

		HRESULT FinalConstruct()
		{
			return profiler.create();
		}
	};
}