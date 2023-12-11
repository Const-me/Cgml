namespace Mistral.Model;
using Cgml;
using SentencePiece;
using System.Collections.Generic;

/// <summary>Wraps SentencePiece library, written by Google and ported to C#</summary>
sealed class Tokenizer: IDisposable, iTokenizer
{
	readonly iProcessor model;
	public readonly int vocabSize;
	readonly int idBOS;
	public readonly int idEOS;
	public readonly int idPad;

	public Tokenizer( in Device dev, Stream stream, int length )
	{
		model = dev.device.loadSentencePieceModel( stream, length );

		var info = model.getInfo();
		vocabSize = info.vocabSize;
		idBOS = info.idBOS;
		idEOS = info.idEOS;
		idPad = info.idPad;

		Logger.Debug( "Created tokenizer, {0} tokens in the vocabulary", vocabSize );
	}

	/// <summary>Given an input string, encode it into a sequence of ids.<br />
	/// Optionally insert BOS at the start, and EOS at the end.</summary>
	public int[] encode( string str, bool bos, bool eos )
	{
		ReadOnlySpan<int> tokens = model.encode( str );

		int i = tokens.Length;
		if( bos ) i++;
		if( eos ) i++;
		int[] arr = new int[ i ];

		i = 0;
		if( bos )
			arr[ i++ ] = idBOS;

		tokens.CopyTo( arr.AsSpan().Slice( i, tokens.Length ) );

		if( eos )
		{
			i += tokens.Length;
			arr[ i ] = idEOS;
		}
		return arr;
	}

	/// <summary>Given a sequence of ids, decodes it into a detokenized output.</summary>
	public string decode( ReadOnlySpan<int> tokens ) =>
		model.decode( tokens );

	public void Dispose()
	{
		model?.Dispose();
	}

	void iTokenizer.encode( List<int> tokens, string text )
	{
		ReadOnlySpan<int> span = model.encode( text );
		foreach( var token in span )
			tokens.Add( token );
	}
	int iTokenizer.idBOS => idBOS;
	int iTokenizer.idEOS => idEOS;
}