namespace MistralChat;
using Cgml;
using Mistral;
using System;
using System.Diagnostics;
using System.IO;
using System.Windows;
using System.Windows.Controls;

/// <summary>Interaction logic for ImportDialog.xaml</summary>
public partial class ImportDialog: Window
{
	/// <summary>Magnet link to MISTRAL AI - 7B - v0.1  27/09/23</summary>
	const string strMagnet = @"magnet:?xt=urn:btih:208B101A0F51514ECF285885A8B0F6FB1A1E4D7D&dn=mistral-7B-v0.1&tr=udp%3a%2f-tp%3a%2f%2f2Ftracker.opentrackr.org%3a1337%2fannounce&tr=https%3a%2f-tps%3a%2f%2ft.co%2fHAadNvH1t0%3a443%2fannounce&tr=wss%3a%2f%2fwstracker.online";

	/// <summary>HTTP link to Instruct version</summary>
	const string strChat = @"https://files.mistral-7b-v0-1.mistral.ai/mistral-7B-instruct-v0.1b.tar";

	public ImportDialog()
	{
		InitializeComponent();

		Compression[] items = compressionItems();
		cbCompression.ItemsSource = items;
		// Preselect BCML1 item in the combobox
		cbCompression.SelectedItem = items[ 1 ];
		lblCompressionInfo.Text = getCompressionInfo();
	}

	public sealed class Compression
	{
		public readonly eTensorLayout layout;
		public readonly string label, desc;
		public override string ToString() => label;

		public Compression( eTensorLayout layout, string label, string desc )
		{
			this.layout = layout;
			this.label = label;
			this.desc = desc;
		}
	}

	Compression[] compressionItems()
	{
		Compression[] res = new Compression[ 2 ];
		res[ 0 ] = new Compression( eTensorLayout.Dense, "Dense FP16", "Dense tensors converted from BF16 to FP16" );
		res[ 1 ] = new Compression( eTensorLayout.BCML1, "BCML1, 4 bits / element", "Custom lossy codec which quantizes weights into 4 bits per element" );
		return res;
	}

	void magnet_Click( object sender, RoutedEventArgs e )
	{
		using var p = Process.Start( new ProcessStartInfo( strMagnet ) { UseShellExecute = true } );
	}
	void chat_Click( object sender, RoutedEventArgs e )
	{
		using var p = Process.Start( new ProcessStartInfo( strChat ) { UseShellExecute = true } );
	}

	void browse_Click( object sender, RoutedEventArgs e )
	{
		FolderPicker picker = new FolderPicker();
		picker.Title = "Select model location";
		bool? res = null;
		try
		{
			res = picker.ShowDialog( this, true );
		}
		catch( Exception ex )
		{
			this.reportError( "Folder picker failed", ex );
		}

		if( true != res )
			return;

		tbSourceFolder.Text = picker.ResultPath;
	}

	MistralDir? selectedDir = null;
	void sourceFolder_TextChanged( object sender, TextChangedEventArgs e )
	{
		string dir = tbSourceFolder.Text;
		MistralDir ld = new MistralDir( dir );
		lblSourceStatus.Text = ld.statusMessage;
		selectedDir = ld;
		lblCompressionInfo.Text = getCompressionInfo();
	}

	void cbCompression_SelectionChanged( object sender, SelectionChangedEventArgs e )
	{
		lblCompressionInfo.Text = getCompressionInfo();
	}

	string getCompressionInfo()
	{
		Compression? c = cbCompression.SelectedItem as Compression;
		if( null == c )
			return "";

		var m = selectedDir;
		if( null == m?.dir )
			return c.desc;

		long modelBytes = m.bytes ?? 0;
		switch( c.layout )
		{
			case eTensorLayout.Dense:
				break;
			case eTensorLayout.BCML1:
				modelBytes = ( modelBytes / ( 2 * 32 ) ) * ( 4 + 16 );
				break;
		}

		return $"{c.desc}; approximate VRAM required: {Cgml.MiscUtils.printMemoryUse( modelBytes )}";
	}

	void inputError( string message )
	{
		MessageBox.Show( this, message, "Unable to import", MessageBoxButton.OK, MessageBoxImage.Information );
	}

	void ok_Click( object sender, RoutedEventArgs e )
	{
		if( null == selectedDir?.dir )
		{
			inputError( "Please select model root folder" );
			return;
		}

		Compression? c = cbCompression.SelectedItem as Compression;
		if( null == c )
		{
			inputError( "Please select tensors compression" );
			return;
		}

		result = new TorchSource
		{
			tokenizer = Path.Combine( selectedDir.dir, MistralDir.tokenizer ),
			weights = selectedDir.dir,
			compression = c.layout
		};
		DialogResult = true;
	}

	/// <summary>After the dialog is closed with the OK button, this property contains the result of the dialog</summary>
	public TorchSource? result { get; private set; }
}