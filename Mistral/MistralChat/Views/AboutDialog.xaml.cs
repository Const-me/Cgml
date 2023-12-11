namespace MistralChat;
using System.Diagnostics;
using System.Reflection;
using System.Windows;

/// <summary>About dialog</summary>
public partial class AboutDialog: Window
{
	public AboutDialog()
	{
		InitializeComponent();

		// Format version label, using the assembly info of this .exe
		string ver = Assembly.GetExecutingAssembly().GetName().Version?.ToString() ?? string.Empty;
		while( ver.Length > 3 && ver.EndsWith( ".0" ) )
			ver = ver.Substring( 0, ver.Length - 2 );
		lblVersion.Text = $"Version {ver}";
	}

	/// <summary>Handler for the source code hyperlink</summary>
	void source_Click( object sender, RoutedEventArgs e )
	{
		const string strSourceRepo = @"https://github.com/Const-me/Cgml";
		using var p = Process.Start( new ProcessStartInfo( strSourceRepo ) { UseShellExecute = true } );
	}

	/// <summary>Handler for the OK button</summary>
	void ok_Click( object sender, RoutedEventArgs e ) =>
		DialogResult = true;
}