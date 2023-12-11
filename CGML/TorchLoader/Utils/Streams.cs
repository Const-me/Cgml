namespace Torch;
using System;
using System.Diagnostics;
using System.IO.Compression;

/// <summary>A small array of open streams</summary>
sealed class Streams: Disposables<Stream>
{
	public Streams( IEnumerable<string> fileNames ) :
		base( fileNames.OrderBy( x => x ).Select( File.OpenRead ) )
	{ }

	public Streams( IEnumerable<ZipArchiveEntry> entries ) :
		base( entries.Select( e => e.Open() ) )
	{ }

	static void read( Stream stream, Span<byte> chunk )
	{
		while( !chunk.IsEmpty )
		{
			int received = stream.Read( chunk );
			if( received <= 0 )
				throw new EndOfStreamException();
			Debug.Assert( received <= chunk.Length );
			chunk = chunk.Slice( received );
		}
	}

	/// <summary>Concatenate tensor data from all streams into a single span</summary>
	public void concatTensors( Span<byte> span, ReadOnlySpan<int> lengths )
	{
		if( lengths.Length != arr.Length )
			throw new ArgumentException();

		for( int i = 0; i < lengths.Length; i++ )
		{
			Stream src = arr[ i ];

			int left = lengths[ i ];
			Span<byte> chunk = span.Slice( 0, left );
			span = span.Slice( left );
			read( src, chunk );
		}

		Debug.Assert( span.IsEmpty );
	}

	/// <summary>Concatenate rows from multiple tensors</summary>
	public void concatRows( Span<byte> span, int countRows, ReadOnlySpan<int> lengths )
	{
		int outputRowLength = 0;
		foreach( int x in lengths )
			outputRowLength += x;
		if( lengths.Length != arr.Length || outputRowLength * countRows != span.Length )
			throw new ArgumentException();

		for( int y = 0; y < countRows; y++ )
		{
			for( int i = 0; i < lengths.Length; i++ )
			{
				int cb = lengths[ i ];
				arr[ i ].Read( span.Slice( 0, cb ) );
				span = span.Slice( cb );
			}
		}

		Debug.Assert( span.IsEmpty );
	}

	/// <summary>Read tensor data from first stream only, read and discard the data from other streams</summary>
	public void copyFirstTensor( Span<byte> span, ReadOnlySpan<int> lengths )
	{
		if( lengths.Length != arr.Length )
			throw new ArgumentException();
#if DEBUG
		Guid? firstHash = null;
#endif
		for( int i = lengths.Length - 1; i >= 0; i-- )
		{
			Stream src = arr[ i ];
			Span<byte> chunk = span.Slice( 0, lengths[ i ] );
			read( src, chunk );
#if DEBUG
			Guid hash = MiscUtils.md5( chunk );
			firstHash ??= hash;
			if( firstHash.Value != hash )
				throw new ApplicationException( "The skipped pieces were different" );
#endif
		}
	}
}