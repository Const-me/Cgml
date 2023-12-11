namespace Cgml;
using Cgml.Internal;
using ComLight;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;

/// <summary>Factory methods implemented by the C++ DLL</summary>
public static class Library
{
	static Library()
	{
		if( Environment.OSVersion.Platform != PlatformID.Win32NT )
			throw new ApplicationException( "This library requires Windows OS" );
		if( !Environment.Is64BitProcess )
			throw new ApplicationException( "This library only works in 64-bit processes" );
		if( RuntimeInformation.ProcessArchitecture != Architecture.X64 )
			throw new ApplicationException( "This library requires a processor with AMD64 instruction set" );
		if( !Sse41.IsSupported )
			throw new ApplicationException( "This library requires a CPU with SSE 4.1 support" );
		NativeLogger.startup();
	}

	const string dll = "Cgml.dll";

	[DllImport( dll, CallingConvention = RuntimeClass.defaultCallingConvention, PreserveSig = false )]
	internal static extern void setupLogger( [In] ref sLoggerSetup setup );

	/// <summary>Set up delegate to receive log messages from the C++ library</summary>
	internal static void setLogSink( eLogLevel lvl, eLoggerFlags flags = eLoggerFlags.SkipFormatMessage, pfnLogMessage? pfn = null )
	{
		NativeLogger.setup( lvl, flags, pfn );
	}

	[DllImport( dll, CallingConvention = RuntimeClass.defaultCallingConvention, PreserveSig = true )]
	static extern int createDeviceAndContext(
		[In] ref sDeviceParams deviceParams,
		[MarshalAs( UnmanagedType.CustomMarshaler, MarshalTypeRef = typeof( Marshaler<iDevice> ) )] out iDevice dev,
		[MarshalAs( UnmanagedType.CustomMarshaler, MarshalTypeRef = typeof( Marshaler<iContext> ) )] out iContext ctx );

	/// <summary>Initialize DirectCompute runtime</summary>
	/// <seealso cref="listGraphicAdapters" />
	public static Device createDevice( sDeviceParams deviceParams )
	{
		NativeLogger.prologue();
		int hr = createDeviceAndContext( ref deviceParams, out iDevice dev, out iContext ctx );
		NativeLogger.throwForHR( hr );
		return new Device( dev, ctx );
	}

	[UnmanagedFunctionPointer( CallingConvention.StdCall )]
	delegate void pfnListAdapters( [In, MarshalAs( UnmanagedType.LPWStr )] string name, IntPtr pv );

	[DllImport( dll, CallingConvention = RuntimeClass.defaultCallingConvention, PreserveSig = true )]
	static extern int listGPUs( [MarshalAs( UnmanagedType.FunctionPtr )] pfnListAdapters pfn, IntPtr pv );

	/// <summary>Enumerate graphics adapters on this computer, and return their names.</summary>
	public static string[] listGraphicAdapters()
	{
		List<string> list = new List<string>();
		pfnListAdapters pfn = delegate ( string name, IntPtr pv )
		{
			Debug.Assert( pv == IntPtr.Zero );
			list.Add( name );
		};

		NativeLogger.prologue();
		int hr = listGPUs( pfn, IntPtr.Zero );
		NativeLogger.throwForHR( hr );

		return list.ToArray();
	}

	[DllImport( dll, CallingConvention = RuntimeClass.defaultCallingConvention, PreserveSig = true )]
	internal static extern int dbgTensorsDiff( out TensorsDiff diff,
		IntPtr a, [In] ref sTensorBuffer aDesc,
		IntPtr b, [In] ref sTensorBuffer bDesc );

	[DllImport( dll, CallingConvention = CallingConvention.StdCall )]
	static extern bool downcastFloats( ref byte buffer, int lengthFloats );

	/// <summary>Downcast F32 to F16 in-place using "round to nearest" mode,<br/>
	/// and return a boolean telling whether all output numbers are exactly equal to the input ones</summary>
	/// <remarks>When this function returns true, no data was lost as the result of the downcasting.<br/>
	/// Requires F16C instruction support in the CPU.</remarks>
	public static bool downcastFloats( Span<byte> buffer )
	{
		int lengthBytes = buffer.Length;
		if( 0 != ( lengthBytes % 4 ) )
			throw new ArgumentException();

		return downcastFloats( ref buffer.GetPinnableReference(), lengthBytes / 4 );
	}

	[DllImport( dll, CallingConvention = CallingConvention.StdCall )]
	static extern bool isAllZero( IntPtr buffer, int length );

	/// <summary>True if the buffer contains at least 1 number not equal to 0.0</summary>
	public static bool isAllZero( ReadOnlySpan<float> buffer )
	{
		unsafe
		{
			fixed ( float* ptr = buffer )
			{
				return isAllZero( (IntPtr)ptr, buffer.Length );
			}
		}
	}
}