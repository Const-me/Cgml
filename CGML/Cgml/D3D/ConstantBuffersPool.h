#pragma once

// Implements a small pool of dynamic constant buffers of varying sizes.
// The buffers are created on-demand.
class ConstantBuffersPool
{
	std::vector<CComPtr<ID3D11Buffer>> pool;

public:
	ConstantBuffersPool( size_t maxVectors = 8 );
	~ConstantBuffersPool() = default;

	HRESULT updateAndBind( ID3D11Device* dev, ID3D11DeviceContext* ctx, const uint8_t* constantBufferData, int cbSize );
};