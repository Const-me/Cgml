#pragma warning disable CS8618, CS0649
namespace Cgml;
using System.IO;
using System.IO.Compression;
using System.Runtime.Serialization;
using System.Xml;

/// <summary>A collection of compiled compute shader binaries</summary>
/// <remarks>When the GPU doesn't have the optional FP64 support, the call to ID3D11Device.CreateComputeShader fails for FP64 shaders.<br/>
/// To workaround, the data structure includes patch tables for FP64-supporting GPUs.</remarks>
[DataContract]
sealed class ShaderPackage
{
	/// <summary>Concatenated shader binaries</summary>
	[DataMember]
	internal byte[] blob;

	/// <summary>Length = count of binaries + 1, value = offset of the start of the binary in the <c>blob</c> array</summary>
	/// <remarks>The last value = length of the <c>blob</c> array</remarks>
	[DataMember]
	internal int[] binaries;

	/// <summary>Binaries to use on GPUs without any optional features</summary>
	[DataMember]
	internal ushort[] shaders;

	/// <summary>Patches for the array of shaders on GPUs with basic FP64 support.<br/>
	/// The array contains serialized pairs of [ shader, binary ] integers</summary>
	[DataMember( EmitDefaultValue = false, IsRequired = false )]
	internal ushort[]? fp1;

	/// <summary>Patches for the array of shaders on GPUs with extended FP64 support.<br/>
	/// The array contains serialized pairs of [ shader, binary ] integers</summary>
	[DataMember( EmitDefaultValue = false, IsRequired = false )]
	internal ushort[]? fp2;

	static IXmlDictionary xmlDictionary() => new PreSharedDictionary(
		"http://schemas.datacontract.org/2004/07/Cgml",
		"http://www.w3.org/2001/XMLSchema-instance",
		"http://schemas.microsoft.com/2003/10/Serialization/Arrays",
		"unsignedShort",
		nameof( ShaderPackage ),
		nameof( blob ),
		nameof( binaries ),
		nameof( shaders ),
		nameof( fp1 ),
		nameof( fp2 )
	);

#if PACK_SHADERS_TOOL
	internal void write( Stream stm )
	{
		using var zip = new GZipStream( stm, CompressionLevel.SmallestSize );
		var serializer = new DataContractSerializer( typeof( ShaderPackage ) );
		using var writer = XmlDictionaryWriter.CreateBinaryWriter( zip, xmlDictionary() );
		serializer.WriteObject( writer, this );
		writer.Flush();
	}
#endif

	/// <summary>Decompress and de-serialize these shaders from the stream</summary>
	public static ShaderPackage read( Stream stm )
	{
		using var zip = new GZipStream( stm, CompressionMode.Decompress );
		var serializer = new DataContractSerializer( typeof( ShaderPackage ) );
		using var reader = XmlDictionaryReader.CreateBinaryReader( zip, xmlDictionary(), XmlDictionaryReaderQuotas.Max );
		ShaderPackage? res = serializer.ReadObject( reader ) as ShaderPackage;
		return res ?? throw new ApplicationException();
	}
}