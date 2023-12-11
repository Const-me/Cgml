namespace Cgml;
using ComLight;
using System.ComponentModel;
using System.Runtime.InteropServices;

/// <summary>Callback to produce new data for a tensor</summary>
[UnmanagedFunctionPointer( CallingConvention.StdCall )]
public delegate int pfnWriteTensorUnsafe( IntPtr rdi, int countBytes, IntPtr pv );

/// <summary>Callback to consume downloaded data for a tensor</summary>
[UnmanagedFunctionPointer( CallingConvention.StdCall )]
public delegate int pfnReadTensorUnsafe( IntPtr rsi, int countBytes, IntPtr pv );

/// <summary>This flag affects what specifically <see cref="iContext.download" /> method does</summary>
public enum eDownloadFlag: byte
{
	/// <summary>Copy from VRAM to output</summary>
	None = 0,
	/// <summary>Copy from VRAM to staging buffer</summary>
	CopyToStaging = 1,
	/// <summary>Copy from staging buffer to output</summary>
	ReadStaging = 2,
}

/// <summary>Represents a device context which generates rendering commands</summary>
[ComInterface( "f249552e-5942-4480-b833-6ba124974f46", eMarshalDirection.ToManaged ), CustomConventions( typeof( Internal.NativeLogger ) )]
public interface iContext: IDisposable
{
	/// <summary>Bind a compute shader, update and bind the constant buffer</summary>
	[EditorBrowsable( EditorBrowsableState.Never )]
	void bindShader( ushort id, IntPtr constantBufferData, int cbSize );

	/// <summary>Dispatch the currently bound compute shader</summary>
	void dispatch( int groupsX, int groupsY = 1, int groupsZ = 1 );

	/// <summary>Bind 1 tensor for writing</summary>
	void bindTensors0( iTensor result );

	/// <summary>Bind 1 tensor for writing, and 1 for reading</summary>
	void bindTensors1( iTensor result, iTensor arg0 );

	/// <summary>Bind 1 tensor for writing, and 2 for reading</summary>
	void bindTensors2( iTensor result, iTensor arg0, iTensor arg1 );

	/// <summary>Bind 1 tensor for writing, and 3 for reading</summary>
	void bindTensors3( iTensor result, iTensor arg0, iTensor arg1, iTensor arg2 );

	/// <summary>Bind 2 tensors for writing</summary>
	void bindTensors2w( iTensor result0, iTensor result1 );

	/// <summary>Bind 2 tensors for writing and 2 for reading</summary>
	void bindTensors2w2r( iTensor res0, iTensor res1, iTensor arg0, iTensor arg1 );

	/// <summary>Bind 2 tensors for writing and 1 for reading</summary>
	void bindTensors2w1r( iTensor res0, iTensor res1, iTensor arg0 );

	/// <summary>Unbind input tensors</summary>
	void unbindInputs();
	/// <summary>Copy the entire contents of the source tensor to the destination tensor using the GPU</summary>
	void copy( iTensor destination, iTensor source );

	/// <summary>Replace tensor data</summary>
	[EditorBrowsable( EditorBrowsableState.Never )]
	void writeDynamic( iTensor tensor, [In] ref TensorShape shape, [MarshalAs( UnmanagedType.FunctionPtr )] pfnWriteTensorUnsafe pfn, IntPtr pv );

	/// <summary>Download tensor data from VRAM to system memory</summary>
	void download( iTensor tensor, [MarshalAs( UnmanagedType.FunctionPtr )] pfnReadTensorUnsafe? pfn, IntPtr pv, eDownloadFlag flag );

	/// <summary>Begin a profiler block</summary>
	void profilerBlockStart( ushort id );

	/// <summary>End the current profiler block</summary>
	void profilerBlockEnd();

	/// <summary>Get the profiler data</summary>
	[EditorBrowsable( EditorBrowsableState.Never )]
	void profilerGetData( [MarshalAs( UnmanagedType.FunctionPtr )] pfnProfilerDataUnsafe pfn, IntPtr pv );

	/// <summary>Create a vector of compute shaders, using selected pieces of the blob</summary>
	[EditorBrowsable( EditorBrowsableState.Never )]
	void createComputeShaders( int length, [In] ref ShaderBinarySlice blobs, [In, MarshalAs( UnmanagedType.LPArray )] byte[] data, int dataSize );

	/// <summary>Write payload data of the tensor to the stream</summary>
	void writeTensorData( iTensor tensor, [WriteStream] Stream stream );
}