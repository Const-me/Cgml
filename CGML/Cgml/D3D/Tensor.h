#pragma once
#include "../API/iTensor.cl.h"
#include "../API/eDownloadFlag.h"
#include "../../ComLightLib/comLightServer.h"
#include "createBuffer.h"

namespace Cgml
{
	class Tensor : public ComLight::ObjectRoot<iTensor>
	{
	protected:
		CComPtr<ID3D11ShaderResourceView> srv;
		sTensorDesc desc;

		HRESULT COMLIGHTCALL getDesc( sTensorDesc& rdi ) const noexcept override final
		{
			rdi = desc;
			return S_OK;
		}

		HRESULT COMLIGHTCALL view( const TensorShape& newShape ) noexcept override final;

		HRESULT COMLIGHTCALL getMemoryUse( __m128i* rdi ) const noexcept override;

		Tensor( const sTensorDesc& d, ID3D11ShaderResourceView* s, ID3D11UnorderedAccessView* u ) :
			srv( s ), desc( d ) { }

	public:
		// Constructor for immutable tensor
		Tensor( const sTensorDesc& d, ID3D11ShaderResourceView* view ) :
			srv( view ), desc( d )
		{
			assert( d.usage == eBufferUse::Immutable );
		}

		const sTensorDesc& getDesc() const { return desc; }

		virtual ID3D11UnorderedAccessView* writeView() const
		{
			return nullptr;
		}
		ID3D11ShaderResourceView* readView() const
		{
			return srv;
		}
		virtual HRESULT download( ID3D11DeviceContext* context, pfnReadTensor pfn, void* pv, eDownloadFlag flag, bool skipWarning = false ) const noexcept;

		HRESULT createImmutableRaw( ID3D11Device* device, const std::vector<uint32_t>& data );

		virtual HRESULT loadData( ID3D11Device* device, const uint8_t* rsi, size_t cb ) noexcept;
	};

	class ResizeableTensor : public Tensor
	{
		ID3D11UnorderedAccessView* writeView() const override
		{
			return uav;
		}

		HRESULT loadData( ID3D11Device* device, const uint8_t* rsi, size_t cb ) noexcept override final;
	protected:
		CComPtr<ID3D11UnorderedAccessView> uav;
		uint32_t capacity = 0;

		virtual HRESULT resizeImpl( ID3D11Device* dev, const BufferDesc& desc ) noexcept;

	public:
		ResizeableTensor( const sTensorDesc& d, ID3D11ShaderResourceView* s, ID3D11UnorderedAccessView* u ) :
			Tensor( d, s, u ), uav( u ),
			capacity( (uint32_t)d.shape.countElements() )
		{ }

		uint32_t getCapacity() const
		{
			return capacity;
		}

		virtual HRESULT replaceData( ID3D11DeviceContext* context, pfnUpdateDynamicTensor pfn, void* pv, uint32_t bytes );

		HRESULT reshape( const TensorShape& shape );

		HRESULT resize( ID3D11Device* dev, const sTensorDesc& desc ) noexcept;
	};

	class DynamicTensor : public ResizeableTensor
	{
		CComPtr<ID3D11Buffer> buffer;

		HRESULT resizeImpl( ID3D11Device* dev, const BufferDesc& desc ) noexcept override final;

		HRESULT COMLIGHTCALL getMemoryUse( __m128i* rdi ) const noexcept override final;

	public:
		DynamicTensor( const sTensorDesc& d, ID3D11ShaderResourceView* view, ID3D11Buffer* buf ) :
			ResizeableTensor( d, view, nullptr ),
			buffer( buf )
		{
			assert( d.usage == eBufferUse::Dynamic );
		}

		HRESULT replaceData( ID3D11DeviceContext* context, pfnUpdateDynamicTensor pfn, void* pv, uint32_t bytes ) override final;

		operator ID3D11Buffer* ( ) const
		{
			return buffer;
		}
	};

	class StagingTensor : public ResizeableTensor
	{
		CComPtr<ID3D11Buffer> bufferVram, bufferStaging;

		HRESULT resizeImpl( ID3D11Device* dev, const BufferDesc& desc ) noexcept override final;

		virtual HRESULT COMLIGHTCALL getMemoryUse( __m128i* rdi ) const noexcept override final;

	public:
		StagingTensor( const sTensorDesc& d, ID3D11ShaderResourceView* s, ID3D11UnorderedAccessView* u, ID3D11Buffer* vram, ID3D11Buffer* staging ) :
			ResizeableTensor( d, s, u ),
			bufferVram( vram ),
			bufferStaging( staging )
		{
			assert( d.usage == eBufferUse::ReadWriteDownload );
		}

		HRESULT replaceData( ID3D11DeviceContext* context, pfnUpdateDynamicTensor pfn, void* pv, uint32_t bytes ) override final;

		HRESULT download( ID3D11DeviceContext* context, pfnReadTensor pfn, void* pv, eDownloadFlag flag, bool skipWarning ) const noexcept override final;
	};
}