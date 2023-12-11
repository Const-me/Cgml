#pragma once

namespace DirectCompute
{
	bool initializeRenderDoc();

	class CaptureRaii
	{
		ID3D11Device* device = nullptr;

	public:
		CaptureRaii( ID3D11Device* dev );
		CaptureRaii( const CaptureRaii& ) = delete;
		~CaptureRaii();
	};
}