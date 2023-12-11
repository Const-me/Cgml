using Cgml;
using Mistral;
using System.Text;
using System.Windows;
using System.Windows.Input;

namespace MistralChat
{
	/// <summary>Interaction logic for ProfilerOutputWindow.xaml</summary>
	public partial class ProfilerOutputWindow: Window
	{
		public ProfilerOutputWindow()
		{
			InitializeComponent();
		}

		static void printEntries( StringBuilder sb, ProfilerData.Entry[] arr )
		{
			sb.Clear();

			foreach( var i in arr )
			{
				sb.Append( i.name );
				sb.Append( '\t' );

				PrintedTime total = i.result.total;
				sb.AppendFormat( "{0} {1}", total.value, total.unit );
				if( i.result.count > 1 )
				{
					PrintedTime avg = i.result.average;
					PrintedTime max = i.result.maximum;
					sb.AppendFormat( ", {0} calls, {1} {2} avg, {3} {4} max",
						i.result.count,
						avg.value, avg.unit,
						max.value, max.unit );
				}
				sb.AppendLine();
			}
		}

		internal ProfilerOutputWindow( ProfilerData data )
		{
			InitializeComponent();
			StringBuilder sb = new StringBuilder();
			printEntries( sb, data.shaders );
			tbShaders.Text = sb.ToString();
		}

		void window_PreviewKeyDown( object sender, System.Windows.Input.KeyEventArgs e )
		{
			if( e.Key == Key.Escape )
				DialogResult = false;
		}
	}
}