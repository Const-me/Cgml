#pragma once
#include <vector>
#include <string>
#include "../../Cgml/API/iSentencePiece.cl.h"
#include "../../ComLightLib/comLightServer.h"
#include "../src/sentencepiece_processor.h"

namespace SentencePiece
{
	// Implementation of that COM interface
	class Processor : public ComLight::ObjectRoot<iProcessor>
	{
		sentencepiece::SentencePieceProcessor proc;
		std::vector<int> m_tokens;
		std::string m_text;

		HRESULT COMLIGHTCALL encode( const char* input, const int** tokens, int& length ) noexcept override final;
		HRESULT COMLIGHTCALL decode( const int* tokens, int count, const char** str ) noexcept override final;
		HRESULT COMLIGHTCALL getInfo( sProcessorInfo& rdi ) const noexcept override final;

	public:

		HRESULT load( ComLight::iReadStream* stream, uint32_t length );
	};
}