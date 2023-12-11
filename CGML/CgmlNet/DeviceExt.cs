﻿namespace Cgml;

/// <summary>Extension methods for <see cref="iDevice" /> COM interface</summary>
public static class DeviceExt
{
	/// <summary>Create immutable tensor in VRAM from the data in system memory</summary>
	public static iTensor uploadImmutableTensor( this iDevice dev, sTensorDesc desc, ReadOnlySpan<byte> data )
	{
		unsafe
		{
			fixed( byte* rsi = data )
				return dev.uploadImmutableTensor( ref desc, (IntPtr)rsi, data.Length );
		}
	}

	/// <summary>Create a dense row major tensor of the specified size</summary>
	/// <remarks>The initial content of the memory for the buffer is undefined.<br />
	/// You need to write the buffer content some other way before the resource is read.</remarks>
	public static iTensor createDense( this iDevice dev, in Int128 size, eDataType dataType = eDataType.FP16 )
	{
		sTensorDesc desc = new sTensorDesc
		{
			shape = new TensorShape( size ),
			dataType = dataType,
			usage = eBufferUse.ReadWrite,
			layout = eTensorLayout.Dense
		};
		return dev.createTensor( ref desc );
	}
}