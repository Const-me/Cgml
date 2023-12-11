namespace Cgml;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

/// <summary>Extension methods for <see cref="iTensor" /> COM interface</summary>
public static class TensorExt
{
	/// <summary>Get description structure of the tensor</summary>
	public static sTensorDesc getDesc( this iTensor tensor )
	{
		tensor.getDesc( out sTensorDesc res );
		return res;
	}

	/// <summary>Get memory usage of the tensor; first ulong value is system RAM, second is VRAM.</summary>
	public static Vector128<long> getMemoryUse( this iTensor? tensor )
	{
		if( tensor != null )
		{
			tensor.getMemoryUse( out Int128 vec );
			return Unsafe.As<Int128, Vector128<long>>( ref vec );
		}
		return Vector128<long>.Zero;
	}

	/// <summary>Get memory usage of the tensor; first ulong value is system RAM, second is VRAM.</summary>
	public static Vector128<long> getMemoryUse( this Tensor? tensor ) =>
		( tensor?.native ).getMemoryUse();

	/// <summary>Get the size of the tensor</summary>
	public static Int128 getSize( this iTensor tensor )
	{
		tensor.getDesc( out sTensorDesc desc );
		return desc.shape.size;
	}

	/// <summary>Count of elements in the tensor</summary>
	public static int countElements( this iTensor tensor )
	{
		tensor.getDesc( out sTensorDesc desc );
		return desc.size.horizontalProduct();
	}
}