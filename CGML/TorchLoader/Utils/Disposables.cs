namespace Torch;
using System;
using System.Collections;

/// <summary>Base class for an array of disposable things</summary>
abstract class Disposables<T>:
	IEnumerable<T>,
	IDisposable
	where T : IDisposable
{
	protected readonly T[] arr;

	public Disposables( IEnumerable<T> sequence )
	{
		List<T> list = new List<T>();
		try
		{
			foreach( T item in sequence )
				list.Add( item );
		}
		catch
		{
			foreach( T item in list )
				item?.Dispose();
			throw;
		}
		arr = list.ToArray();
	}

	public void Dispose()
	{
		foreach( T item in arr )
			item?.Dispose();
	}
	public IEnumerator<T> GetEnumerator()
	{
		return ( (IEnumerable<T>)arr ).GetEnumerator();
	}
	IEnumerator IEnumerable.GetEnumerator()
	{
		return arr.GetEnumerator();
	}
	public T this[ int index ] => arr[ index ];
	public int length => arr.Length;
}