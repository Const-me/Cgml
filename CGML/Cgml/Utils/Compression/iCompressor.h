#pragma once
#include "../../API/iTensor.cl.h"
#include <memory>

// API for BCML compressor, implemented on top of the Windows built-in thread pool
struct iCompressor
{
	// Get buffer for source data of a tensor
	// When available, the function returns a recycled buffer, to save some malloc() calls
	virtual HRESULT getBuffer( std::vector<__m256i>& vec, size_t lengthBytes ) = 0;

	// Launch a new job to compress the tensor
	virtual HRESULT bcml( Cgml::iTensor** rdi, const Cgml::sTensorDesc& desc, std::vector<__m256i>& data, size_t lengthBytes ) = 0;

	// Wait for all pending jobs to finish
	virtual HRESULT join() = 0;

	virtual ~iCompressor() {}

	// Create the tensor compressor
	static HRESULT create( std::unique_ptr<iCompressor>& rdi, ID3D11Device* device );
};