This project compiles all compute shaders which implement GPU-running pieces of the Mistral ML model inference.

And in a post-build step, it runs the PackShaders.exe tool to pack compiled shaders,
and generate C# boilerplate code and data structures to dispatch them.