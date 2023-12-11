namespace Torch;
using System.Collections;
using System.IO.Compression;

/// <summary>Unpickles the metadata portion of the model</summary>
public static class MetadataLoader
{
	/// <summary>Unpickle the metadata portion of the model</summary>
	static Dictionary<string, Tensor> load( Reader unpickler, Stream stm )
	{
		var obj = unpickler.load( stm );
		if( obj is Tensor tt )
		{
			// This happens for Pickle files produced with `torch.save` Python API, when input is a single tensor
			return new Dictionary<string, Tensor>() { { "", tt } };
		}

		if( obj is not Hashtable table )
			throw new ArgumentException();

		// The serialization order is random, reorder tensors alphabetically instead
		List<string> keys = table.Keys
			.Cast<string>()
			.OrderBy( k => k )
			.ToList();
		Dictionary<string, Tensor> dict = new Dictionary<string, Tensor>( keys.Count );

		foreach( string key in keys )
		{
			Tensor val = table[ key ] as Tensor ?? throw new ArgumentException();
			dict.Add( key, val );
		}
		return dict;
	}

	/// <summary>Load metadata from a single ZIP archive</summary>
	public static Dictionary<string, Tensor> load( ZipArchive zip, string entryName )
	{
		Reader unpickler = new Reader();
		ZipArchiveEntry e = zip.GetEntry( entryName ) ?? throw new ArgumentException( $"ZIP entry is missing: {entryName}" );
		using var stm = e.Open();
		return load( unpickler, stm );
	}

	/// <summary>Load metadata from a collection of ZIP archives</summary>
	public static IEnumerable<Dictionary<string, Tensor>> load( IEnumerable<(ZipArchive, string)> source )
	{
		Reader unpickler = new Reader();
		foreach( (ZipArchive zip, string entryName) in source )
		{
			ZipArchiveEntry e = zip.GetEntry( entryName ) ?? throw new ArgumentException( $"ZIP entry is missing: {entryName}" );
			using var stm = e.Open();
			yield return load( unpickler, stm );
		}
	}
}