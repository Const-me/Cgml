namespace SentencePiece;
using ComLight;
using System.ComponentModel;
using System.Runtime.InteropServices;

/// <summary>A sane API of the Google’s SentencePiece C++ library</summary>
/// <remarks>The interface is not thread safe: the returned tokens/strings are stored in the memory owned by the object.</remarks>
[ComInterface( "d045f91d-b65e-4cca-b372-e49545eab55a", eMarshalDirection.ToManaged ), CustomConventions( typeof( Cgml.Internal.NativeLogger ) )]
public interface iProcessor: IDisposable
{
	/// <summary>Given an input string, encode it into a sequence of ids.</summary>
	[EditorBrowsable( EditorBrowsableState.Never )]
	void encode( [In, MarshalAs( UnmanagedType.LPUTF8Str )] string input, out IntPtr tokens, out int length );

	/// <summary>Given a sequence of ids, decodes it into a detokenized output.</summary>
	[EditorBrowsable( EditorBrowsableState.Never )]
	void decode( IntPtr tokens, int length, out IntPtr text );

	/// <summary>Get some information about the model</summary>
	[RetValIndex]
	sProcessorInfo getInfo();
}