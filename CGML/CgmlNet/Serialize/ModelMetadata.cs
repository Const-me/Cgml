namespace Cgml.Serialize;
using System.Runtime.Serialization;

/// <summary>This object is serialized into <c>metadata.bin</c> entry in the archives</summary>
[DataContract]
sealed class ModelMetadata
{
	[DataMember]
	public int formatVersion;

	[DataMember]
	public string? modelType;
}