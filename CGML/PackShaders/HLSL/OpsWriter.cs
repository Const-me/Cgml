﻿namespace PackShaders;

sealed class OpsWriter: IDisposable
{
	StreamWriter writer;
	public OpsWriter()
	{
		string path = Path.Combine( Program.inputs.result, "ContextOps.cs" );
		writer = File.CreateText( path );
		writer.WriteLine( @"// This source file is generated by PackShaders.exe tool, from HLSL source codes
namespace {0};
using Cgml;

static class ContextOps
{{", Program.inputs.ns );
	}

	public void Dispose()
	{
		writer?.Write( "}" );
		writer?.Flush();
		writer?.Dispose();
	}

	static void ensureSequentialSlots( ResourceBinding[] arr )
	{
		for( int i = 0; i < arr.Length; i++ )
			if( i != arr[ i ].slot )
				throw new ApplicationException( "Resource slots must be sequential, from zero" );
	}

	bool first = true;

	public void add( string shaderName, ShaderReflection hlsl )
	{
		ResourceBinding[] uav = hlsl.bindings
			.Where( b => b.kind == eResourceKind.UAV )
			.OrderBy( b => b.slot )
			.ToArray();

		ResourceBinding[] srv = hlsl.bindings
			.Where( b => b.kind == eResourceKind.SRV )
			.OrderBy( b => b.slot )
			.ToArray();

		if( uav.Length < 1 )
			throw new ArgumentException();

		ensureSequentialSlots( uav );
		ensureSequentialSlots( srv );

		if( this.first )
			this.first = false;
		else
			writer.WriteLine();

		if( null != hlsl.comment )
			writer.WriteLine( "\t/// <summary>{0}</summary>", hlsl.comment );
		writer.Write( "\tpublic static void {0}( this iContext ctx, ConstantBuffers.{0} cb", shaderName );
		foreach( var b in uav.Concat( srv ) )
			writer.Write( ", iTensor {0}", b.name );
		writer.WriteLine( " )" );
		writer.WriteLine( "\t{" );
		writer.WriteLine( "\t\tctx.bindShader( (ushort)eShader.{0}, ref cb );", shaderName );
		string methodName;
		if( uav.Length == 1 )
			methodName = $"bindTensors{srv.Length}";
		else if( uav.Length == 2 )
		{
			methodName = srv.Length switch
			{
				0 => "bindTensors2w",
				1 => "bindTensors2w1r",
				2 => "bindTensors2w2r",
				_ => throw new NotImplementedException()
			};
		}
		else
			throw new NotImplementedException();

		writer.Write( "\t\tctx.{0}(", methodName );

		bool first = true;
		foreach( var b in uav.Concat( srv ) )
		{
			if( first )
			{
				first = false;
				writer.Write( ' ' );
			}
			else
				writer.Write( ", " );
			writer.Write( b.name );
		}

		writer.WriteLine( " );" );
		writer.WriteLine( "\t}" );
	}
}