namespace Cgml;
using System.Runtime.CompilerServices;

/// <summary>Utility class to produce log messages from C#, and forward them into the provided destination.</summary>
/// <remarks>Messages printed by C++ backend also go there</remarks>
public static class Logger
{
	static pfnLogMessage? sink = null;

	/// <summary>Error message</summary>
	public static void Error( string message )
	{
		sink?.Invoke( eLogLevel.Error, message );
	}
	/// <summary>Warning message</summary>
	public static void Warning( string message )
	{
		sink?.Invoke( eLogLevel.Warning, message );
	}
	/// <summary>Informational message</summary>
	public static void Info( string message )
	{
		sink?.Invoke( eLogLevel.Info, message );
	}
	/// <summary>Debug message</summary>
	public static void Debug( string message )
	{
		sink?.Invoke( eLogLevel.Debug, message );
	}

	static eLogLevel level = eLogLevel.Warning;

	[MethodImpl( MethodImplOptions.AggressiveInlining )]
	static bool willLog( this eLogLevel lvl ) =>
		(byte)lvl <= (byte)level;

	/// <summary>Informational message</summary>
	public static void Info( string message, params object[] args )
	{
		if( eLogLevel.Info.willLog() )
			sink?.Invoke( eLogLevel.Info, string.Format( message, args ) );
	}

	/// <summary>Debug message</summary>
	public static void Debug( string message, params object[] args )
	{
		if( eLogLevel.Debug.willLog() )
			sink?.Invoke( eLogLevel.Debug, string.Format( message, args ) );
	}

	/// <summary>Initialize logging</summary>
	public static void setup( eLogLevel lvl, pfnLogMessage? pfn = null, eLoggerFlags flags = eLoggerFlags.None )
	{
		sink = pfn;
		level = lvl;
		Library.setLogSink( lvl, flags, sink );
	}
}