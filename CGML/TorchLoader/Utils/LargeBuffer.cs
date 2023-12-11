namespace Torch;
using System.Runtime.InteropServices;

/// <summary>Large buffer used to merge tensor from chunks</summary>
/// <remarks>The size of that buffer exceeds 300 MB.<br />
/// This class bypasses .NET garbage collector, uses unmanaged allocator instead</remarks>
[StructLayout( LayoutKind.Auto )]
struct LargeBuffer: IDisposable
{
	IntPtr data;
	int capacity;

	public Span<byte> resize( int length )
	{
		if( capacity < length )
		{
			if( data != IntPtr.Zero )
			{
				Marshal.FreeHGlobal( data );
				data = IntPtr.Zero;
			}
			capacity = 0;

			data = Marshal.AllocHGlobal( length );
			if( data == IntPtr.Zero )
				throw new OutOfMemoryException();
			capacity = length;
		}

		unsafe
		{
			return new Span<byte>( (byte*)data, length );
		}
	}

	public void Dispose()
	{
		if( data != IntPtr.Zero )
		{
			Marshal.FreeHGlobal( data );
			data = IntPtr.Zero;
		}
		capacity = 0;
	}

	public UnmanagedMemoryStream readStream( int length )
	{
		if( data == IntPtr.Zero )
			throw new ArgumentException( "The buffer has no data" );
		if( length <= 0 || length > capacity )
			throw new ArgumentOutOfRangeException();

		unsafe
		{
			byte* pointer = (byte*)data;
			return new UnmanagedMemoryStream( pointer, length, length, FileAccess.Read );
		}
	}
}