namespace Torch;
using Cgml;
using Razorvine.Pickle.Objects;

/// <summary>Tensor data, as loaded from the ZIP</summary>
public sealed class Tensor
{
	/// <summary>Storage information</summary>
	public readonly Storage storage;
	/// <summary>No idea what this means</summary>
	public readonly int offset;
	/// <summary>Shape of the tensor</summary>
	public readonly TensorShape shape;
	/// <summary>No idea what this means</summary>
	public readonly bool requires_grad;

	/// <summary>The constructor needs to be public because otherwise Pickle can't create the object with Activator.CreateInstance</summary>
	public Tensor( Storage storage, int offset, object[] size, object[] stride, bool requires_grad, ClassDict metadata )
	{
		this.storage = storage;
		this.offset = offset;
		shape = TensorShape.unpickle( size, stride );
		this.requires_grad = requires_grad;
	}
}