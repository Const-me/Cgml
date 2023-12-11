namespace Cgml.Serialize;
using System.IO.Compression;
using System.Runtime.InteropServices;
using System.Runtime.Serialization;

/// <summary>Abstract base class for both surrogate providers</summary>
/// <remarks>This rarely-used API allows to substitute types while serializing.<br />
/// This is exactly what we need to serialize tensors in VRAM</remarks>
abstract class SurrogateProvider: ISerializationSurrogateProvider
{
	Type ISerializationSurrogateProvider.GetSurrogateType( Type type )
	{
		if( type == typeof( iTensor ) )
			return typeof( TensorMetadata );
		return type;
	}

	public virtual object GetObjectToSerialize( object obj, Type targetType )
	{
		throw new NotImplementedException();
	}

	public virtual object GetDeserializedObject( object obj, Type targetType )
	{
		throw new NotImplementedException();
	}

	/// <summary>ZIP archives in .NET standard library can’t read nor write multiple entries in parallel.<br />
	/// For this reason, when saving or loading models we postpone saving/loading tensor payloads.<br />
	/// This structure keeps such postponed tensor.</summary>
	[StructLayout( LayoutKind.Auto )]
	protected readonly struct TensorEntry
	{
		/// <summary>0-based index of the tensor, it translates into the name of the ZIP entry</summary>
		public int id { get; init; }
		/// <summary>The tensor being saved or loaded</summary>
		public iTensor tensor { get; init; }
		/// <summary>Payload size in bytes</summary>
		public int byteWidth { get; init; }
	}

	protected readonly List<TensorEntry> entries = new List<TensorEntry>();

	/// <summary>Generate a function to map 0-based tensor ID into ZIP entry name</summary>
	protected Func<int, string> entryNameFunc( string subfolder )
	{
		int length = (int)Math.Ceiling( Math.Log10( entries.Count ) );
		return ( int i ) =>
		{
			string index = i.ToString().PadLeft( length, '0' );
			return $"{subfolder}/{index}.bin";
		};
	}

	/// <summary>1.0 / ( total payload size of all pending tensors)</summary>
	protected double progressMultiplier()
	{
		long cb = 0;
		foreach( var i in entries )
			cb += i.byteWidth;
		return 1.0 / cb;
	}
}

/// <summary>Surrogate provider to save models</summary>
/// <remarks>It assigns 0-based integer IDs to the tensors being saved, and collects these tensors in the list</remarks>
sealed class SaveProvider: SurrogateProvider
{
	int tensorId = 0;

	public override object GetObjectToSerialize( object obj, Type targetType )
	{
		if( obj is iTensor tensor )
		{
			TensorMetadata res = new TensorMetadata( tensor, ref tensorId );
			entries.Add( new TensorEntry { id = res.id, tensor = tensor, byteWidth = res.byteWidth } );
			return res;
		}
		return obj;
	}

	/// <summary>Call this method after serialized the model metadata, to write tensors payloads into different ZIP entries</summary>
	public void writePayload( iContext context, ZipArchive zip, string subfolder, bool compressWeights, Action<double>? pfnProgress )
	{
		if( entries.Count <= 0 )
			return;
		var makeName = entryNameFunc( subfolder );

		double progressMul = 0;
		if( null != pfnProgress )
			progressMul = progressMultiplier();

		CompressionLevel compressionLevel = compressWeights ? Serializer.compressionLevel : CompressionLevel.NoCompression;

		long bytesWritten = 0;
		foreach( TensorEntry i in entries )
		{
			string entryName = makeName( i.id );
			var e = zip.CreateEntry( entryName, compressionLevel );
			using var stm = e.Open();
			context.writeTensorData( i.tensor, stm );

			if( null != pfnProgress )
			{
				bytesWritten += i.byteWidth;
				pfnProgress( progressMul * bytesWritten );
			}
		}
	}
}

/// <summary>Surrogate provider to save models</summary>
/// <remarks>It creates uninitialized tensors while loading metadata,
/// and collects these tensors in the list</remarks>
sealed class LoadProvider: SurrogateProvider
{
	readonly iDevice device;

	public LoadProvider( iDevice device )
	{
		this.device = device;
	}

	public override object GetDeserializedObject( object obj, Type targetType )
	{
		if( obj is TensorMetadata meta )
		{
			sTensorDesc desc = new sTensorDesc
			{
				shape = meta.shape,
				dataType = (eDataType)meta.dataType,
				usage = (eBufferUse)meta.usage,
				layout = (eTensorLayout)meta.layout,
			};
			iTensor res = device.createUninitializedTensor( ref desc );
			entries.Add( new TensorEntry { id = meta.id, tensor = res, byteWidth = meta.byteWidth } );
			return res;
		}
		return obj;
	}

	/// <summary>Call this method after de-serialized the model metadata, to load tensors payloads from different ZIP entries</summary>
	public void readPayload( ZipArchive zip, string subfolder, Action<double>? pfnProgress )
	{
		if( entries.Count <= 0 )
			return;
		var makeName = entryNameFunc( subfolder );

		double progressMul = 0;
		if( null != pfnProgress )
			progressMul = progressMultiplier();

		long bytesRead = 0;
		foreach( TensorEntry i in entries )
		{
			string entryName = makeName( i.id );
			var e = zip.GetEntry( entryName ) ??
				throw new ArgumentException( $"ZIP entry is missing: {entryName}" );
			using var stm = e.Open();
			device.loadTensor( i.tensor, stm, (int)e.Length );

			if( null != pfnProgress )
			{
				bytesRead += e.Length;
				pfnProgress( progressMul * bytesRead );
			}
		}
	}
}