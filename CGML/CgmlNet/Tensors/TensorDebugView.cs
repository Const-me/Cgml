namespace Cgml;

/// <summary>This class is only used by Visual Studio IDE, and only while debugging.<br />
/// It implements a better visualizer for <see cref="iTensor" /> COM interfaces.</summary>
sealed class TensorDebugView
{
	/// <summary>Query and cache some info about the tensor</summary>
	public TensorDebugView( iTensor it )
	{
		var desc = it.getDesc();
		shape = desc.shape;
		dataType = desc.dataType;
		usage = desc.usage;
		layout = desc.layout;
		memoryUsed = MiscUtils.printMemoryUse( it.getMemoryUse() );
	}

	/// <summary>Size and stride</summary>
	public readonly TensorShape shape;

	/// <summary>Type of elements</summary>
	public readonly eDataType dataType;

	/// <summary>Usage flags for the GPU buffer</summary>
	public readonly eBufferUse usage;

	/// <summary>VRAM layout of the tensor</summary>
	public readonly eTensorLayout layout;

	/// <summary>Memory usage summary for the tensor</summary>
	public readonly string memoryUsed;
}