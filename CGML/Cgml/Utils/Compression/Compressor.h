#pragma once
#include <mutex>
#include <condition_variable>
#include <optional>
#include "iCompressor.h"
#include "../../D3D/Tensor.h"

class Compressor: public iCompressor
{
	PTP_WORK work = nullptr;
	CComPtr<ID3D11Device> device;
	std::mutex mutex;
	std::condition_variable condVar;

	std::vector<std::vector<__m256i>> poolInputs;
	std::vector<std::vector<uint32_t>> poolCompressed;

	HRESULT getBuffer( std::vector<__m256i>& vec, size_t lengthBytes ) noexcept override final;

	HRESULT bcml( Cgml::iTensor** rdi, const Cgml::sTensorDesc& desc, std::vector<__m256i>& data, size_t lengthBytes ) noexcept override final;

	HRESULT join() noexcept override final;

	static void __stdcall workCallbackStatic( PTP_CALLBACK_INSTANCE Instance, PVOID Context, PTP_WORK Work );

	HRESULT workCallback();

	volatile HRESULT m_status = S_FALSE;

	struct PendingJob
	{
		ComLight::CComPtr<Cgml::Tensor> tensor;
		std::vector<__m256i> sourceVector;
		std::array<uint32_t, 4> sourceStride;
		Cgml::eDataType sourceType;
	};
	std::optional<PendingJob> pending;

	struct CompleteJob
	{
		ComLight::CComPtr<Cgml::Tensor> tensor;
		std::vector<uint32_t> bufferData;
	};
	std::vector<CompleteJob> completeTensors;

	HRESULT compressImpl( PendingJob& job, std::vector<uint32_t>& resultBuffer );

	HRESULT uploadCompleteTensors();

public:

	Compressor( ID3D11Device* dev ) : device( dev ) { }

	~Compressor() override;

	HRESULT create();
};