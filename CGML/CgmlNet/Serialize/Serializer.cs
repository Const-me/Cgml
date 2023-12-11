namespace Cgml.Serialize;
using System.IO.Compression;
using System.Runtime.Serialization;
using System.Xml;

/// <summary>Utility class to serialize models in ZIP archives</summary>
public sealed class Serializer
{
	const string modelEntry = "model.bin";
	const string metadataEntry = "metadata.bin";
	const string tensorSubfolder = "Tensors";

	readonly Type modelType;
	readonly DataContractSerializer dcsModel;
	readonly DataContractSerializer dcsMetadata;
	readonly IXmlDictionary xmlDictionary;

	string[] builtinDictionary() => new string[]
	{
		"http://www.w3.org/2001/XMLSchema-instance",
		"http://schemas.datacontract.org/2004/07/Cgml",
		"http://schemas.datacontract.org/2004/07/Cgml.Serialize",
		"type",
		"x", "y", "z", "w",
		nameof(TensorMetadata),
		nameof(TensorMetadata.id),
		nameof(TensorMetadata.shape),
		nameof(TensorMetadata.dataType),
		nameof(TensorMetadata.usage),
		nameof(TensorMetadata.layout),
		nameof(TensorShape.size),
		nameof(TensorShape.stride),
		nameof(ModelMetadata),
		nameof(ModelMetadata.formatVersion),
		nameof(ModelMetadata.modelType),
	};

	/// <summary>Compression level, for both metadata and payload</summary>
	/// <remarks>Payload compression is optional, controlled with a boolean passed to <see cref="write" /> method.<br/>
	/// In practice, weight compression seems borderline useless. It saves couple percents of model size, at the cost of an order of magnitude slow down.</remarks>
	internal const CompressionLevel compressionLevel = CompressionLevel.SmallestSize;

	/// <summary>Construct serializer to save and load models of the specified type</summary>
	public Serializer( Type modelType, string[]? extraDictionaryEntries = null )
	{
		this.modelType = modelType;

		Type[] knownTypes = new Type[ 1 ]
		{
			typeof( TensorMetadata ),
		};
		dcsModel = new DataContractSerializer( modelType, knownTypes );

		string[] dict = builtinDictionary();
		if( extraDictionaryEntries?.Length > 0 )
		{
			int lenBuiltin = dict.Length;
			Array.Resize( ref dict, lenBuiltin + extraDictionaryEntries.Length );
			Array.Copy( extraDictionaryEntries, 0, dict, lenBuiltin, extraDictionaryEntries.Length );
		}

		xmlDictionary = new PreSharedDictionary( dict );

		dcsMetadata = new DataContractSerializer( typeof( ModelMetadata ) );
	}

	void serialize( ZipArchive zip, string entry, object obj, DataContractSerializer dcs )
	{
		var e = zip.CreateEntry( entry, compressionLevel );
		using var stm = e.Open();
		using var writer = XmlDictionaryWriter.CreateBinaryWriter( stm, xmlDictionary );
		dcs.WriteObject( writer, obj );
		writer.Flush();
	}

	/// <summary>Serialize the model to the ZIP archive</summary>
	public void write( ZipArchive zip, object model, iContext context, bool compressWeights, Action<double>? pfnProgress )
	{
		if( zip.Mode != ZipArchiveMode.Create )
			throw new ArgumentException( "Unexpected ZIP archive mode" );

		if( model.GetType() != modelType )
			throw new ArgumentException( "Unexpected model type" );

		SaveProvider sp = new SaveProvider();
		dcsModel.SetSerializationSurrogateProvider( sp );

		try
		{
			serialize( zip, modelEntry, model, dcsModel );
		}
		finally
		{
			dcsModel.SetSerializationSurrogateProvider( null );
		}

		ModelMetadata mm = new ModelMetadata
		{
			formatVersion = 1,
			modelType = modelType.FullName
		};
		serialize( zip, metadataEntry, mm, dcsMetadata );

		// throw new ApplicationException( "Saved metadata" );
		sp.writePayload( context, zip, tensorSubfolder, compressWeights, pfnProgress );
	}

	object deserialize( ZipArchive zip, string entry, DataContractSerializer dcs )
	{
		var e = zip.GetEntry( entry ) ??
			throw new ArgumentException( $"ZIP entry missing: {entry}" );

		using var stm = e.Open();
		using var reader = XmlDictionaryReader.CreateBinaryReader( stm, xmlDictionary, XmlDictionaryReaderQuotas.Max );

		return dcs.ReadObject( reader ) ??
			throw new ApplicationException( $"Unable to de-serialize {entry}" );
	}

	/// <summary>De-serialize model from ZIP archive</summary>
	public object read( ZipArchive zip, iDevice device, Action<double>? pfnProgress )
	{
		if( zip.Mode != ZipArchiveMode.Read )
			throw new ArgumentException( "Unexpected ZIP archive mode" );

		ModelMetadata mm = (ModelMetadata)deserialize( zip, metadataEntry, dcsMetadata );
		if( mm.formatVersion != 1 )
			throw new ArgumentException( $"Unsupported format version {mm.formatVersion}" );
		if( mm.modelType != modelType.FullName )
			throw new ArgumentException( $"Unexpected type of the model: expected {modelType.FullName}, got {mm.modelType}" );

		object result;
		LoadProvider lp = new LoadProvider( device );
		dcsModel.SetSerializationSurrogateProvider( lp );
		try
		{
			result = deserialize( zip, modelEntry, dcsModel );
		}
		finally
		{
			dcsModel.SetSerializationSurrogateProvider( null );
		}
		lp.readPayload( zip, tensorSubfolder, pfnProgress );

		return result;
	}
}