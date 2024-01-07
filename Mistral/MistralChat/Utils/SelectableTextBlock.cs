// Disable nullable-related warnings 
#pragma warning disable 8600, 8604, 8601, 8618, 8602
namespace MistralChat;
using System.Reflection;
using System.Windows;
using System.Windows.Controls;

// Initially copy-pasted from there: https://stackoverflow.com/a/45627524/126995
sealed class TextEditorWrapper
{
	// TODO [very low, performance]: instead of caching types and properties from reflection,
	// use System.Linq.Expressions to compile delegates, and cache these compiled delegates instead.
	// Not sure that gonna work for internal classes and properties from another assembly, but it might.

	static readonly Type TextEditorType;
	static readonly MethodInfo RegisterMethod;
	static readonly PropertyInfo TextContainerProp;
	static readonly PropertyInfo IsReadOnlyProp;
	static readonly PropertyInfo TextViewProp;
	static readonly PropertyInfo TextContainerTextViewProp;

	static TextEditorWrapper()
	{
		TextEditorType = Type.GetType( "System.Windows.Documents.TextEditor, PresentationFramework" );

		IsReadOnlyProp = TextEditorType.GetProperty( "IsReadOnly", BindingFlags.Instance | BindingFlags.NonPublic );
		TextViewProp = TextEditorType.GetProperty( "TextView", BindingFlags.Instance | BindingFlags.NonPublic );
		RegisterMethod = TextEditorType.GetMethod( "RegisterCommandHandlers",
			BindingFlags.Static | BindingFlags.NonPublic, null, new[] { typeof( Type ), typeof( bool ), typeof( bool ), typeof( bool ) }, null );

		Type TextContainerType = Type.GetType( "System.Windows.Documents.ITextContainer, PresentationFramework" );
		TextContainerTextViewProp = TextContainerType.GetProperty( "TextView" );

		TextContainerProp = typeof( TextBlock ).GetProperty( "TextContainer", BindingFlags.Instance | BindingFlags.NonPublic );
	}

	public static void RegisterCommandHandlers( Type controlType, bool acceptsRichContent, bool readOnly, bool registerEventListeners )
	{
		RegisterMethod.Invoke( null, new object[] { controlType, acceptsRichContent, readOnly, registerEventListeners } );
	}

	TextEditorWrapper( object textContainer, FrameworkElement uiScope, bool isUndoEnabled )
	{
		_editor = Activator.CreateInstance( TextEditorType, BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.CreateInstance,
			null, new[] { textContainer, uiScope, isUndoEnabled }, null );
	}

	readonly object _editor;

	public static TextEditorWrapper CreateFor( TextBlock tb )
	{
		var textContainer = TextContainerProp.GetValue( tb );

		TextEditorWrapper editor = new TextEditorWrapper( textContainer, tb, false );
		IsReadOnlyProp.SetValue( editor._editor, true );
		TextViewProp.SetValue( editor._editor, TextContainerTextViewProp.GetValue( textContainer ) );

		return editor;
	}
}

public sealed class SelectableTextBlock: TextBlock
{
	static SelectableTextBlock()
	{
		FocusableProperty.OverrideMetadata( typeof( SelectableTextBlock ), new FrameworkPropertyMetadata( true ) );
		TextEditorWrapper.RegisterCommandHandlers( typeof( SelectableTextBlock ), true, true, true );
		// Remove focus rectangle around the control
		FocusVisualStyleProperty.OverrideMetadata( typeof( SelectableTextBlock ), new FrameworkPropertyMetadata( (object)null ) );
	}

	readonly TextEditorWrapper _editor;

	public SelectableTextBlock()
	{
		_editor = TextEditorWrapper.CreateFor( this );
	}
}