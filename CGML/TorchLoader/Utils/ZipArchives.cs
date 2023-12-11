namespace Torch;
using System;
using System.IO.Compression;

/// <summary>A small array of ZIP archives, opened for reading</summary>
sealed class ZipArchives: Disposables<ZipArchive>
{
	public ZipArchives( IEnumerable<Stream> stm ) :
		base( stm.Select( s => new ZipArchive( s, ZipArchiveMode.Read ) ) )
	{ }

	public T[] deserialize<T>( Func<int, string> entryName, Func<Stream, T> read )
	{
		T[] res = new T[ arr.Length ];

		for( int i = 0; i < arr.Length; i++ )
		{
			string name = entryName( i );
			ZipArchiveEntry e = arr[ i ].GetEntry( name ) ??
				throw new ArgumentException( $"ZIP entry missing: {name}" );
			using var stm = e.Open();
			res[ i ] = read( stm );
		}

		return res;
	}
}