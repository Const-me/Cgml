namespace Cgml;

/// <summary>Utility class to consume messages from <see cref="Logger" />, and print them to console.</summary>
public static class ConsoleLogger
{
	static readonly object syncRoot = new object();

	static readonly ConsoleColor[] colors = new ConsoleColor[ 4 ]
	{
		ConsoleColor.Red,
		ConsoleColor.Yellow,
		ConsoleColor.Green,
		ConsoleColor.Blue
	};

	/// <summary>This function goes into the log consumer delegate of the <see cref="Logger" /> class</summary>
	static void print( eLogLevel level, string message )
	{
		lock( syncRoot )
		{
			ConsoleColor pc = Console.ForegroundColor;
			Console.ForegroundColor = colors[ (byte)level ];
			Console.WriteLine( message );
			Console.ForegroundColor = pc;
		}
	}

	/// <summary>Setup console logging</summary>
	public static void setup( eLogLevel lvl, eLoggerFlags flags = eLoggerFlags.None )
	{
		Logger.setup( lvl, print, flags );
	}
}