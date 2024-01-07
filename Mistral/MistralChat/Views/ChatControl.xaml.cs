namespace MistralChat;
using MistralChat.ViewModels;
using System;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;

/// <summary>Interaction logic for ChatControl.xaml</summary>
public partial class ChatControl: UserControl
{
	public ChatControl()
	{
		InitializeComponent();
	}

	async void generate( iChatVM vm, string text )
	{
		try
		{
			tbInput.IsEnabled = false;
			tbInitialPrompt.IsEnabled = false;

			await vm.generate( text );
		}
		catch( Exception ex )
		{
			Window wnd = Window.GetWindow( this );
			wnd.reportError( "Unable to generate the response", ex );
		}
		finally
		{
			tbInput.IsEnabled = true;
			tbInitialPrompt.IsEnabled = true;
			tbInput.Focus();
		}
	}

	iChatVM? viewModel => DataContext as iChatVM;

	void chat_KeyDown( object sender, KeyEventArgs e )
	{
		if( e.Key != Key.Enter )
			return;
		if( !e.KeyboardDevice.Modifiers.HasFlag( ModifierKeys.Control ) )
			return;

		// Detected Ctrl+Enter keyboard event
		string text = tbInput.Text;
		tbInput.Text = "";

		if( string.IsNullOrWhiteSpace( text ) )
			return;

		iChatVM? vm = viewModel;
		if( null == vm )
			return;
		generate( vm, text );
	}

	void copyChat_Click( object sender, RoutedEventArgs e )
	{
		iChatVM? vm = viewModel;
		if( null == vm )
			return;

		StringBuilder sb = new StringBuilder();
		foreach( iChatMessageVM msg in vm.history )
		{
			if( msg.user )
				sb.Append( "User: " );
			else
				sb.Append( "AI: " );
			sb.AppendLine( msg.text );
		}

		Clipboard.SetText( sb.ToString() );
	}

	void copyMessage_Click( object sender, RoutedEventArgs e )
	{
		// Cast sender to FrameworkElement, in fact it's a MenuItem but we only need a FrameworkElement base class here
		FrameworkElement? fwe = sender as FrameworkElement;
		// Get the right clicked chat message. Luckily, menu items inherit data context from the owner.
		iChatMessageVM? message = fwe?.DataContext as iChatMessageVM;
		if( null != message )
		{
			// Copy the message to clipboard
			Clipboard.SetText( message.text );
		}
	}

	void regenerate_click( object sender, RoutedEventArgs e )
	{
		iChatVM? vm = viewModel;
		if( null == vm )
			return;
		if( vm.history.Count < 2 )
			return;

		if( vm.isDeterministic )
		{
			var wnd = Window.GetWindow( this );
			MessageBox.Show( wnd, @"Please uncheck “Options / Disable randomness”.
While that option is checked, regeneration gonna produce the same response you already see.",
				"Disabled with an option", MessageBoxButton.OK, MessageBoxImage.Information );
			return;
		}

		string text = vm.history[ vm.history.Count - 2 ].text;
		vm.history.RemoveAt( vm.history.Count - 1 );
		vm.history.RemoveAt( vm.history.Count - 1 );
		generate( vm, text );
	}

	void clear_click( object sender, RoutedEventArgs e ) =>
		viewModel?.history.Clear();

	void cancel_click( object sender, RoutedEventArgs e ) =>
		viewModel?.cancel();
}