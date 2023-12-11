namespace Mistral.Model;
using Cgml;
using System;
using System.Diagnostics;
using System.Runtime.InteropServices;

/// <summary>Rotated + permuted view of a tensor</summary>
struct RotatedTensorShape
{
	struct Dim
	{
		public int stride;
		public WrappedRange range;
	}
	Dim dim0, dim1, dim2, dim3;

	public static RotatedTensorShape createDense( in TensorShape shape )
	{
		RotatedTensorShape res;
		res.dim0 = new Dim { stride = shape.stride.x, range = WrappedRange.single( shape.size.x ) };
		res.dim1 = new Dim { stride = shape.stride.y, range = WrappedRange.single( shape.size.y ) };
		res.dim2 = new Dim { stride = shape.stride.z, range = WrappedRange.single( shape.size.z ) };
		res.dim3 = new Dim { stride = shape.stride.w, range = WrappedRange.single( shape.size.w ) };
		return res;
	}

	public void unwrapZ( in WrappedRange range )
	{
		if( dim2.range.isWrapped || dim2.range.offset0 != 0 )
			throw new NotSupportedException();
		if( range.length > dim2.range.length )
			throw new ArgumentOutOfRangeException();
		dim2.range = range;
	}

	public void permute( byte x, byte y, byte z, byte w )
	{
		if( ( x | y | z | w ) >= 4 )
			throw new ArgumentOutOfRangeException( "Permutation indices must be in [ 0 .. 3 ] range each" );
		if( x == y || x == z || x == w || y == z || y == w || z == w )
			throw new ArgumentException( "Permutation indices must be unique" );

		RotatedTensorShape source = this;
		ReadOnlySpan<RotatedTensorShape> spanShape = MemoryMarshal.CreateReadOnlySpan( ref source, 1 );
		ReadOnlySpan<Dim> span = MemoryMarshal.Cast<RotatedTensorShape, Dim>( spanShape );
		Debug.Assert( span.Length == 4 );

		dim0 = span[ x ];
		dim1 = span[ y ];
		dim2 = span[ z ];
		dim3 = span[ w ];
	}

	/// <summary>Size of the unrotated tensor</summary>
	public Int128 size =>
		new Int128( dim0.range.length, dim1.range.length, dim2.range.length, dim3.range.length );

	/// <summary>Create a constant buffer for the unrotate compute shader</summary>
	public ConstantBuffers.unrotate unrotateConstants( in TensorShape resultShape )
	{
		Debug.Assert( resultShape.size == size );
		Debug.Assert( resultShape.stride.x == 1 );

		ConstantBuffers.unrotate res;
		res.inputStrides = new Int128( dim0.stride, dim1.stride, dim2.stride, dim3.stride );
		res.width = (uint)resultShape.size.x;
		res.outputStrides = resultShape.stride.yzw;
		res.inputOffset0 = new Int128( dim0.range.offset0, dim1.range.offset0, dim2.range.offset0, dim3.range.offset0 );
		res.inputLength0 = new Int128( dim0.range.length0, dim1.range.length0, dim2.range.length0, dim3.range.length0 );
		res.inputOffset1 = new Int128( dim0.range.inputOffset1, dim1.range.inputOffset1, dim2.range.inputOffset1, dim3.range.inputOffset1 );
		return res;
	}
}