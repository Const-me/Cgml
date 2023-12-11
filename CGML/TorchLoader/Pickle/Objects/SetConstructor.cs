/* part of Pickle, by Irmen de Jong (irmen@razorvine.net) */

namespace Razorvine.Pickle.Objects;
using System.Collections;

/// <summary>This object constructor creates sets. (HashSet&lt;object&gt;)</summary>
internal class SetConstructor: IObjectConstructor
{
	public object construct( object[] args )
	{
		// create a HashSet, args=arraylist of stuff to put in it
		ArrayList elements = (ArrayList)args[ 0 ];
		IEnumerable<object> array = elements.ToArray();
		return new HashSet<object>( array );
	}
}