namespace MistralChat;
using Microsoft.Win32;
using Mistral;
using System;
using System.Runtime.CompilerServices;

/// <summary>Utility class to save and load preferences stored in Windows registry</summary>
static class Preferences
{
	const string keyPath = @"SOFTWARE\const.me\MistralChat";
	static readonly RegistryKey key;

	static Preferences()
	{
		key = Registry.CurrentUser.CreateSubKey( keyPath, true );
	}

	public static string[]? recentModels
	{
		get
		{
			object? obj = key.GetValue( nameof( recentModels ) );
			if( obj is string[] arr )
				return arr;
			if( obj is string s )
				return new string[ 1 ] { s };
			return null;
		}
		set
		{
			if( null == value || value.Length == 0 )
				key.DeleteValue( nameof( recentModels ), false );
			else
				key.SetValue( nameof( recentModels ), value, RegistryValueKind.MultiString );
		}
	}

	static bool loadBoolean( [CallerMemberName] string? name = null )
	{
		object? obj = key.GetValue( name );
		if( obj is int i )
			return i != 0;
		return false;
	}

	static void storeBoolean( bool val, [CallerMemberName] string? name = null )
	{
		if( val )
			key.SetValue( name, 1, RegistryValueKind.DWord );
		else
			key.DeleteValue( name ?? throw new ApplicationException(), false );
	}

	public static bool loadLastModel
	{
		get => loadBoolean();
		set => storeBoolean( value );
	}

	public static bool disableRandomness
	{
		get => loadBoolean();
		set => storeBoolean( value );
	}

	const string samplingParams = "samplingParams";

	/// <summary>Save sampling parameters to Windows registry</summary>
	/// <remarks>Both floats are packed into a single uint64 registry value</remarks>
	public static void saveSampling( SamplingParams sp )
	{
		unchecked
		{
			ulong low = BitConverter.SingleToUInt32Bits( sp.temperature );
			ulong high = BitConverter.SingleToUInt32Bits( sp.topP );
			ulong val = low | ( high << 32 );
			key.SetValue( samplingParams, val, RegistryValueKind.QWord );
		}
	}

	/// <summary>Load sampling parameters from Windows registry</summary>
	/// <remarks>Returns null when the setting is missing</remarks>
	public static SamplingParams? tryLoadSampling()
	{
		object? obj = key.GetValue( samplingParams );
		if( null == obj )
			return null;

		ulong val;
		if( obj is long i64 )
			val = unchecked((ulong)i64);
		else if( obj is ulong u64 )
			val = u64;
		else
			return null;

		float temperature = BitConverter.UInt32BitsToSingle( (uint)( val & uint.MaxValue ) );
		float topP = BitConverter.UInt32BitsToSingle( (uint)( val >> 32 ) );
		return new SamplingParams( temperature, topP );
	}

	static string? loadString( [CallerMemberName] string? name = null ) =>
		key.GetValue( name ) as string;
	static void storeString( string? val, [CallerMemberName] string? name = null )
	{
		if( null != val )
			key.SetValue( name, val, RegistryValueKind.String );
		else
			key.DeleteValue( name ?? throw new ApplicationException(), false );
	}

	/// <summary>GPU selection preference</summary>
	public static string? gpu
	{
		get => loadString();
		set => storeString( value );
	}

	static int loadInteger( int def, [CallerMemberName] string? name = null )
	{
		object? obj = key.GetValue( name );
		if( null == obj )
			return def;
		if( obj is int i32 )
			return i32;
		if( obj is uint u32 )
			return unchecked((int)u32);
		return def;
	}

	static void storeInteger( int val, int def, [CallerMemberName] string? name = null )
	{
		if( val != def )
			key.SetValue( name, unchecked((uint)val), RegistryValueKind.DWord );
		else
			key.DeleteValue( name ?? throw new ApplicationException(), false );
	}

	/// <summary>GPU queue length preference</summary>
	public static int gpuQueueDepth
	{
		get => loadInteger( Cgml.sDeviceParams.defaultQueueDepth );
		set => storeInteger( value, Cgml.sDeviceParams.defaultQueueDepth );
	}

	public static bool gpuPowerSaver
	{
		get => loadBoolean();
		set => storeBoolean( value );
	}

	/// <summary>Load or save all D3D11 device parameters in 1 shot</summary>
	public static Cgml.sDeviceParams deviceParameters
	{
		get
		{
			var flags = Cgml.sDeviceParams.eDeviceFlags.None;
			if( gpuPowerSaver )
				flags |= Cgml.sDeviceParams.eDeviceFlags.PowerSaver;
			return new Cgml.sDeviceParams( gpu, gpuQueueDepth, flags );
		}
	}

	public static bool gpuHighPerformance
	{
		get => loadBoolean();
		set => storeBoolean( value );
	}
}