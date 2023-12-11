namespace SentencePiece;
using System.Runtime.InteropServices;

/// <summary>Extension methods for <see cref="iProcessor" /></summary>
public static class ProcessorExt
{
	/// <summary>Given an input string, encode it into a sequence of ids.</summary>
	public static ReadOnlySpan<int> encode( this iProcessor processor, string input )
	{
		processor.encode( input, out IntPtr ptr, out int len );
		if( len > 0 )
		{
			unsafe
			{
				return new ReadOnlySpan<int>( (void*)ptr, len );
			}
		}
		else
			return ReadOnlySpan<int>.Empty;
	}

	/// <summary>Given a sequence of ids, decodes it into a detokenized output.</summary>
	public static string decode( this iProcessor processor, ReadOnlySpan<int> tokens )
	{
		IntPtr text;
		unsafe
		{
			fixed( int* ptr = tokens )
			{
				processor.decode( (IntPtr)ptr, tokens.Length, out text );
			}
		}
		return Marshal.PtrToStringUTF8( text ) ?? string.Empty;
	}
}