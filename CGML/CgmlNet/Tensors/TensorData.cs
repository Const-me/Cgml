namespace Cgml;
using System.Runtime.InteropServices;

/// <summary>Tensor data downloaded from VRAM for QA purposes</summary>
public readonly struct TensorData
{
	/// <summary>Describes size and memory layout of a tensor in VRAM</summary>
	public readonly sTensorDesc desc;

	/// <summary>Payload data of the tensor, in managed memory</summary>
	public readonly Array data;

	/// <summary>Create the structure</summary>
	public TensorData( in sTensorDesc desc, Array data )
	{
		this.desc = desc;
		this.data = data;
	}

	ReadOnlySpan<byte> getBytes()
	{
		if( desc.layout != eTensorLayout.Dense )
			throw new NotImplementedException();

		switch( desc.dataType )
		{
			case eDataType.FP16:
			case eDataType.BF16:
				return MemoryMarshal.Cast<ushort, byte>( (ushort[])data );
			case eDataType.FP32:
				return MemoryMarshal.Cast<float, byte>( (float[])data );
			case eDataType.U32:
				return MemoryMarshal.Cast<uint, byte>( (uint[])data );
		}
		throw new NotImplementedException();
	}

	/// <summary>Verify the tensors contain the same data</summary>
	public TensorsDiff diff( in TensorData that )
	{
		if( desc.shape.size != that.desc.shape.size )
			throw new ArgumentException( "Size is different" );
		if( data.Length != that.data.Length )
			throw new ArgumentException( "Array length is different" );

		ReadOnlySpan<byte> s0 = getBytes();
		ReadOnlySpan<byte> s1 = that.getBytes();

		return TensorsDiff.compute( s0, desc, s1, that.desc );
	}

	/// <summary>Save payload data, without any headers</summary>
	public void save( string path )
	{
		using var file = File.Create( path );
		file.Write( getBytes() );
	}
}