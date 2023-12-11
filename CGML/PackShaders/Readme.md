This program is a command line tool which packs compiled compute shaders,
also generates C# boilerplate code and data structures to dispatch them with `CgmlNet` library.

The tool is designed to be launched automagically from the project which compiles these shaders, as a post-build step.

Here's an example from `MistralShaders.vcxproj` project file:

```
<Target Name="PackShaders" AfterTargets="Build">
  <Exec
    Command="$(SolutionDir)CGML/PackShaders/bin/$(Configuration)/PackShaders.exe"
    EnvironmentVariables="SHADERS_PROJECT=&quot;$(ProjectDir)$(ProjectFileName)&quot;;SHADERS_CONFIG=$(Configuration);SHADERS_TEMP=&quot;$(OutDir)&quot;;SHADERS_RESULT=&quot;$(ProjectDir)../MistralModel/Model/Generated/&quot;;SHADERS_NAMESPACE=Mistral.Model" />
</Target>
```

As you see, the inputs are passed in these special environment variables.

All variables are required.

* `SHADERS_PROJECT` — full path to the `*.vcxproj` file which compiles the shaders
* `SHADERS_CONFIG` — configuration, either "Debug" or "Release".
* `SHADERS_TEMP` — folder to search for the compiled shaders, i.e. the output directory of the shaders project .<br/>
Also, to minimize the time it takes to build the project, this tool writes a small binary file `packedShadersHash.bin` to that folder,
which contains 16 bytes of the MD5 of all inputs. When all inputs are the same, the tool exits silently, without writing any files.
* `SHADERS_RESULT` — folder to write results.
* `SHADERS_NAMESPACE` — namespace to use for generated C# codes.

The tool generates 4 output files:

1. `Shaders$(Configuration).bin` contains shader binaries, and a few pieces of metadata to create the shaders.<br/>
Specifically, the file contains an instance of `Cgml.ShaderPackage` C# class serialized into [binary XML](https://learn.microsoft.com/en-us/openspecs/windows_protocols/mc-nbfx/94c66ea1-e79a-4364-af88-1fa7fef2cc33), then compressed with [GZip](https://learn.microsoft.com/en-us/dotnet/api/system.io.compression.gzipstream?view=net-6.0).
2. `eShader.cs` contains a C# enum which identifies a compute shader.<br/>
The name of the enum fields matches the names of the source HLSL files.
3. `ConstantBuffers.cs` contains a C# static class with nested structures.<br/>
The name of these nested structures matches the fields of the `eShader` enum.<br/>
These structures are generated from HLSL source codes to match memory layout of the constant buffer in the slot #0 of these compute shaders.
4. `ContextOps.cs` contains a C# static class with extension methods for `Cgml.iContext` COM interface.<br/>
Names of these extension methods also match the fields of the `eShader` enum.<br/>
These static methods bind the correct shader, bind the constant buffer, bind inputs and outputs, but they don't actually dispatch the shader.

The generated `ContextOps` methods do not dispatch shaders,
because it’s hard to detect correct count of thread groups from HLSL source code.<br/>
Some shaders need 1 thread group per row of the output. Other shaders need 1 thread group per fixed-size block of the output.
Others are doing something else entirely.

Still, despite being technically incomplete, this compile-time code generator helps tremendously.<br/>
It enabled development of complicated GPGPU algorithms with relative ease, and most importantly without compiling a single line of C++.