#pragma once
#include "../../ComLightLib/comLightCommon.h"
#include "../../ComLightLib/streams.h"

namespace SentencePiece
{
	struct sProcessorInfo
	{
		int vocabSize;
		int idBOS;
		int idEOS;
		int idPad;
	};

	// A sane API of the Google’s SentencePiece C++ library
	struct DECLSPEC_NOVTABLE iProcessor : public ComLight::IUnknown
	{
		DEFINE_INTERFACE_ID( "d045f91d-b65e-4cca-b372-e49545eab55a" );

		virtual HRESULT COMLIGHTCALL encode( const char* input, const int** tokens, int& length ) = 0;
		virtual HRESULT COMLIGHTCALL decode( const int* tokens, int count, const char** str ) = 0;
		virtual HRESULT COMLIGHTCALL getInfo( sProcessorInfo& rdi ) const = 0;
	};

	HRESULT loadSentencePieceModel( iProcessor** result, ComLight::iReadStream* stream, uint32_t length );
}