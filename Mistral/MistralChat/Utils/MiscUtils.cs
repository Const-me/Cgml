namespace MistralChat;
using System;
using System.ComponentModel;
using System.Windows;

static class MiscUtils
{
	public static bool isDesignTime =>
		LicenseManager.UsageMode == LicenseUsageMode.Designtime;

	public static void reportError( this Window window, string message, Exception? ex )
	{
		string msg = ( null != ex ) ? $"{message}\n{ex.Message}" : message;
		MessageBox.Show( window, msg, "Operation Failed", MessageBoxButton.OK, MessageBoxImage.Warning );
	}

	public static string pluralString( this int count, string single )
	{
		if( 1 != count )
		{
			if( single[ single.Length - 1 ] != 's' )
				return $"{count} {single}s";
			return $"{count} {single}es";
		}
		return $"1 {single}";
	}
}