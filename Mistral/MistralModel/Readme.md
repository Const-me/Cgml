This project builds a .NET DLL which implements Mistral ML model inference.

It depends on CgmlNet for low-level GPGPU stuff, and exposes an idiomatic C# API.

That API handles following main use cases:

* Allows users to load, save and import these Mistral models.<br/>
I have only tested with the originally released models, however custom fine-tuned models might work too.<br/>
See `Mistral.ModelLoader` class for specific methods.
* Allows to generate text, and implement chat UX.<br/>
See `Mistral.iModel` interface for specific methods.