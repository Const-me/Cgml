namespace Cgml.Serialize;
using System.Runtime.Serialization;

/// <summary>The custom serializer substitutes tensors with these objects while saving,<br />
/// and re-creates tensors from these things while loading</summary>
[DataContract]
sealed record class TensorMetadata
{
	/// <summary>0-based index of the payload data of the tensor</summary>
	[DataMember]
	public readonly int id;

	[DataMember]
	public readonly TensorShape shape;

	// DataContractSerializer outputs strings for enum types.
	// Not what we want, that's why these fields are bytes instead.
	[DataMember]
	public readonly byte dataType, usage, layout;

	internal TensorMetadata( iTensor tensor, ref int id )
	{
		this.id = id++;
		sTensorDesc desc = tensor.getDesc();
		shape = desc.shape;
		dataType = (byte)desc.dataType;
		usage = (byte)desc.usage;
		layout = (byte)desc.layout;
	}

	internal int byteWidth => computeByteWidth();

	int computeByteWidth()
	{
		eTensorLayout layout = (eTensorLayout)this.layout;
		if( layout == eTensorLayout.Dense )
		{
			eDataType dataType = (eDataType)this.dataType;
			return shape.countElements() * dataType.elementSize();
		}
		else
		{
			// Compressed tensors are using byte address buffers in VRAM
			// The strides in the metadata are already expressed in bytes
			return shape.stride.w * shape.size.w;
		}
	}
}