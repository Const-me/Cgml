namespace Torch;
using Cgml;

/// <summary>Implement this abstract class to specify what happens to the tensors being loaded from Python format</summary>
public abstract class LoadTraits
{
	/// <summary>Override this method to selectively specify VRAM layout for these tensors</summary>
	public virtual eTensorLayout tensorVramLayout( string key ) =>
		eTensorLayout.Dense;

	/// <summary>Override this method to specify load transformation for the tensor in the model</summary>
	public virtual eLoadTransform tensorLoadTransform( eDataType storedType, string key ) => eLoadTransform.None;

	/// <summary>Merge action to combine tensors from different ZIP archives</summary>
	/// <remarks>Llama-13B model contains 2 of them, Llama-30B model contains 4</remarks>
	public enum eMergeTactic: byte
	{
		/// <summary>Concatenate tensor data from multiple ZIP archive</summary>
		ConcatData,
		/// <summary>Concatenate each row from multiple ZIP archive</summary>
		ConcatRows,
		/// <summary>Load tensor data from the first ZIP archive in the package</summary>
		/// <remarks>Debug build of this DLL verifies the tensors from different ZIP archives contain identical data,<br />
		/// by computing and comparing MD5 hashes of these tensors.</remarks>
		UseFirst,
		/// <summary>Skip the tensor</summary>
		Ignore,
	}

	/// <summary>Merge action to combine tensors from different ZIP archives</summary>
	public virtual eMergeTactic tensorMergeTactic( string key, ReadOnlySpan<TensorShape> shapes ) =>
		throw new NotImplementedException();
}