namespace Cgml;
using Cgml.Internal;
using System.Diagnostics;
using System.Runtime.InteropServices;

/// <summary>A higher-level callback to produce new data for a tensor</summary>
public delegate void pfnWriteTensor<T>( Span<T> data ) where T : unmanaged;

/// <summary>A higher-level callback to consume tensor data downloaded from VRAM</summary>
public delegate void pfnReadTensor<T>( ReadOnlySpan<T> data ) where T : unmanaged;

/// <summary>Extension methods for <see cref="iContext" /> COM interface</summary>
public static class ContextExt
{
	static int writeImpl<T>( IntPtr rdi, int countBytes, IntPtr pv ) where T : unmanaged
	{
		try
		{
			GCHandle gch = GCHandle.FromIntPtr( pv );
			pfnWriteTensor<T> pfn = ( gch.Target as pfnWriteTensor<T> ) ?? throw new InvalidOperationException();
			int cbElement = Marshal.SizeOf<T>();
			int countElements = countBytes / cbElement;
			unsafe
			{
				Span<T> span = new Span<T>( (void*)rdi, countElements );
				pfn( span );
				return 0;
			}
		}
		catch( Exception ex )
		{
			NativeLogger.captureException( ex );
			return ex.HResult;
		}
	}

	/// <summary>Replace tensor data</summary>
	public static void writeDynamic<T>( this iContext context, iTensor tensor, TensorShape shape, pfnWriteTensor<T> pfn ) where T : unmanaged
	{
		GCHandle gch = GCHandle.Alloc( pfn, GCHandleType.Normal );
		try
		{
			pfnWriteTensorUnsafe wtu = writeImpl<T>;
			context.writeDynamic( tensor, ref shape, wtu, GCHandle.ToIntPtr( gch ) );
		}
		finally
		{
			gch.Free();
		}
	}

	static int downloadImpl<T>( IntPtr rsi, int countBytes, IntPtr pv ) where T : unmanaged
	{
		try
		{
			GCHandle gch = GCHandle.FromIntPtr( pv );
			pfnReadTensor<T> pfn = ( gch.Target as pfnReadTensor<T> ) ?? throw new InvalidOperationException();
			int cbElement = Marshal.SizeOf<T>();
			int countElements = countBytes / cbElement;
			unsafe
			{
				ReadOnlySpan<T> span = new ReadOnlySpan<T>( (void*)rsi, countElements );
				pfn( span );
				return 0;
			}
		}
		catch( Exception ex )
		{
			NativeLogger.captureException( ex );
			return ex.HResult;
		}
	}

	/// <summary>Download tensor from VRAM, feed to the callback</summary>
	public static void download<T>( this iContext context, iTensor tensor, pfnReadTensor<T> pfn, eDownloadFlag flag = eDownloadFlag.None ) where T : unmanaged
	{
		GCHandle gch = GCHandle.Alloc( pfn, GCHandleType.Normal );
		try
		{
			pfnReadTensorUnsafe rtu = downloadImpl<T>;
			context.download( tensor, rtu, GCHandle.ToIntPtr( gch ), flag );
		}
		finally
		{
			gch.Free();
		}
	}

	/// <summary>Bind a compute shader, update and bind the constant buffer</summary>
	public static void bindShader<T>( this iContext context, ushort id, ref T cbData ) where T : unmanaged
	{
		unsafe
		{
			int len = sizeof( T );
			fixed( T* ptr = &cbData )
				context.bindShader( id, (IntPtr)ptr, len );
		}
	}

	/// <summary>Bind a compute shader, update and bind the constant buffer</summary>
	public static void bindShader<T>( this iContext context, ushort id, ReadOnlySpan<T> cbData ) where T : unmanaged
	{
		unsafe
		{
			int len = sizeof( T ) * cbData.Length;
			fixed( T* ptr = cbData )
				context.bindShader( id, (IntPtr)ptr, len );
		}
	}

	/// <summary>Bind a compute shader, update and bind the constant buffer</summary>
	public static void bindShader<T>( this iContext context, ushort id, Span<T> cbData ) where T : unmanaged
	{
		ReadOnlySpan<T> rs = cbData;
		context.bindShader( id, rs );
	}

	static Array downloadTensorData<E>( iContext context, iTensor tensor ) where E : unmanaged
	{
		E[]? arr = null;
		pfnReadTensor<E> pfn = delegate ( ReadOnlySpan<E> data )
		{
			arr = data.ToArray();
		};
		context.download( tensor, pfn, eDownloadFlag.None );
		return arr ?? throw new ApplicationException();
	}

	/// <summary>Download tensor data from VRAM</summary>
	/// <remarks>The implementation is not terribly efficient, implemented for debugging purposes.</remarks>
	public static TensorData downloadTensor( this iContext context, iTensor tensor )
	{
		var desc = tensor.getDesc();
		if( desc.layout != eTensorLayout.Dense )
			throw new NotImplementedException();

		switch( desc.dataType )
		{
			case eDataType.FP16:
			case eDataType.BF16:
				return new TensorData( desc, downloadTensorData<ushort>( context, tensor ) );
			case eDataType.FP32:
				return new TensorData( desc, downloadTensorData<float>( context, tensor ) );
		}
		throw new NotImplementedException();
	}

	/// <summary>Get the profiler data</summary>
	public static ProfilerResult[]? profilerGetData( this iContext context )
	{
		ProfilerResult[]? result = null;
		pfnProfilerDataUnsafe pfn = delegate ( ProfilerResult[] arr, int length, IntPtr pv )
		{
			result = arr;
			return 0;
		};
		context.profilerGetData( pfn, IntPtr.Zero );
		return result;
	}

	/// <summary>Begin a profiler block</summary>
	public static ProfilerBlock profilerBlock( this iContext context, ushort id ) =>
		new ProfilerBlock( context, id );

	/// <summary>Copy tensor data from VRAM into staging buffer of <see cref="eBufferUse.ReadWriteDownload" /> tensor, and leave it there</summary>
	/// <remarks>The method is an asynchronous call</remarks>
	public static void copyToStaging( this iContext context, iTensor tensor )
	{
		Debug.Assert( tensor.getDesc().usage == eBufferUse.ReadWriteDownload );
		context.download( tensor, null, IntPtr.Zero, eDownloadFlag.CopyToStaging );
	}
}