namespace MistralChat.ViewModels;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Threading.Tasks;
using System.Windows;

/// <summary>Interface for the chat control to access the view model</summary>
public interface iChatVM: INotifyPropertyChanged
{
	string initialPrompt { get; set; }

	/// <summary>Chat log</summary>
	ObservableCollection<iChatMessageVM> history { get; }

	/// <summary>Watermark message to show in the input, when the input text is empty</summary>
	string watermark { get; }

	/// <summary>Generate response from the user’s message</summary>
	Task generate( string text );

	bool canRegenerate { get; }

	/// <summary>True if user has checked "Disable randomness" option</summary>
	bool isDeterministic { get; }

	bool canCancel { get; }
	void cancel();

	Visibility visInitialPrompt { get; }
}