#include <assert.h>
#include <array>
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <winerror.h>
#include "Processor.h"
#include <sentencepiece_model.pb.h>

extern "C"
{
	// SentencePiece code fails to compile with C++/20
	// Logger.h from Cgml project uses char8_t type which fails to compile with C++/17
	// Luckily, this hack seems to work, MSVC linker appears to successfully resolve that function,
	// despite the prototype is technically incompatible
	void logError( const char* pszFormat, ... );
}

HRESULT SentencePiece::loadSentencePieceModel( iProcessor** result, ComLight::iReadStream* stream, uint32_t length )
{
	if( nullptr == stream || nullptr == result )
		return E_POINTER;

	ComLight::CComPtr<ComLight::Object<Processor>> obj;
	CHECK( ComLight::Object<Processor>::create( obj ) );
	CHECK( obj->load( stream, length ) );
	obj.detach( result );
	return S_OK;
}

using namespace SentencePiece;
using sentencepiece::util::Status;

static const std::array<HRESULT, 17> knownCodes =
{
	S_OK,
	HRESULT_FROM_WIN32( ERROR_CANCELLED ),	// kCancelled = 1
	E_UNEXPECTED,	// kUnknown = 2
	E_INVALIDARG,	// kInvalidArgument = 3
	HRESULT_FROM_WIN32( ERROR_TIMEOUT ),// kDeadlineExceeded = 4
	HRESULT_FROM_WIN32( ERROR_FILE_NOT_FOUND ),	// kNotFound = 5
	HRESULT_FROM_WIN32( ERROR_ALREADY_EXISTS ),	// kAlreadyExists = 6
	HRESULT_FROM_WIN32( ERROR_ACCESS_DENIED ),	// kPermissionDenied = 7
	E_FAIL,	// kResourceExhausted = 8
	E_FAIL,	// kFailedPrecondition = 9
	E_FAIL,	// kAborted = 10
	DISP_E_OVERFLOW,	// kOutOfRange = 11
	E_NOTIMPL,	// kUnimplemented = 12
	HRESULT_FROM_WIN32( ERROR_INTERNAL_ERROR ),	// kInternal = 13,
	E_FAIL,	// kUnavailable = 14,
	E_FAIL,	// kDataLoss = 15,
	HRESULT_FROM_WIN32( ERROR_NOT_AUTHENTICATED ),	// kUnauthenticated = 16,
};

static HRESULT failedStatus( const Status& status )
{
	assert( !status.ok() );

	logError( "SentencePiece error: %s", status.message() );

	const int code = (int)status.code();
	if( code >= 0 && code < knownCodes.size() )
		return knownCodes[ code ];
	return E_FAIL;
}

HRESULT Processor::load( ComLight::iReadStream* stream, uint32_t length )
{
	std::vector<char> content;
	try
	{
		content.resize( length );
	}
	catch( const std::bad_alloc& )
	{
		return E_OUTOFMEMORY;
	}

	CHECK( stream->read( content ) );

	std::string_view view{ content.data(), content.size() };
	Status s = proc.LoadFromSerializedProto( view );
	if( s.ok() )
		return S_OK;

	return failedStatus( s );
}

HRESULT COMLIGHTCALL Processor::encode( const char* input, const int** tokens, int& length ) noexcept
{
	if( nullptr == input || nullptr == tokens )
		return E_POINTER;
	*tokens = nullptr;
	length = 0;

	m_tokens.clear();
	Status s = proc.Encode( input, &m_tokens );
	if( s.ok() )
	{
		*tokens = m_tokens.empty() ? nullptr : m_tokens.data();
		length = (int)m_tokens.size();
		return S_OK;
	}
	return failedStatus( s );
}

HRESULT COMLIGHTCALL Processor::decode( const int* tokens, int count, const char** str ) noexcept
{
	if( nullptr == tokens || nullptr == str )
		return E_POINTER;

	m_tokens.assign( tokens, tokens + count );
	m_text.clear();
	Status s = proc.Decode( m_tokens, &m_text );
	if( s.ok() )
	{
		*str = m_text.c_str();
		return S_OK;
	}
	*str = nullptr;
	return failedStatus( s );
}

HRESULT COMLIGHTCALL Processor::getInfo( sProcessorInfo& rdi ) const noexcept
{
	rdi.vocabSize = proc.model_proto().pieces_size();
	rdi.idBOS = proc.bos_id();
	rdi.idEOS = proc.eos_id();
	rdi.idPad = proc.pad_id();
	return S_OK;
}