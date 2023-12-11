namespace Torch;
using Cgml;
using System.IO.Compression;
using System.Runtime.InteropServices;

/// <summary>Load ZIP files produced by <c>torch.save</c> in Python</summary>
public static class TensorLoader
{
	static TensorData loadImpl<E>( Tensor src, Stream payload ) where E : unmanaged
	{
		int elts = src.shape.countElements();
		if( elts * Marshal.SizeOf<E>() != payload.Length )
			throw new ArgumentException( "Incorrect payload size" );

		E[] arr = new E[ elts ];
		Span<byte> bytes = MemoryMarshal.Cast<E, byte>( arr );
		payload.ReadExactly( bytes );

		sTensorDesc desc = new sTensorDesc
		{
			shape = src.shape,
			dataType = src.storage.dataType,
			usage = eBufferUse.Immutable,
			layout = eTensorLayout.Dense
		};

		return new TensorData( desc, arr );
	}

	/// <summary>Load ZIP file with a single tensor, produced by <c>torch.save</c> Python code</summary>
	/// <remarks>Returns tensor data in system memory, not in GPU</remarks>
	public static TensorData load( string path )
	{
		using var zipFile = File.OpenRead( path );
		using var zip = new ZipArchive( zipFile, ZipArchiveMode.Read );

		string name = Path.GetFileNameWithoutExtension( path );
		string entry = $"{name}/data.pkl";
		Dictionary<string, Tensor> dict = MetadataLoader.load( zip, entry );

		Tensor src = dict.Values.First();
		entry = $"{name}/data/{src.storage.payload}";
		var e = zip.GetEntry( entry ) ?? throw new ApplicationException( "Tensor data not found" );
		using var sourceStream = e.Open();
		switch( src.storage.dataType )
		{
			case eDataType.FP16:
				return loadImpl<ushort>( src, sourceStream );
			case eDataType.FP32:
				return loadImpl<float>( src, sourceStream );
			default:
				throw new NotImplementedException();
		}
	}

	/// <summary>Create an object to load serialized models to VRAM of the specific adapter</summary>
	public static iWeightsLoader createLoader( iDevice device, LoadTraits traits ) =>
		new LoaderImpl( device, traits );
}