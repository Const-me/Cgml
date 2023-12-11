This software includes non-trivial amount of open-source dependencies.

* Mistral model implementation is technically a port [from Python](https://github.com/mistralai/mistral-src), Apache-2.0 license.<br/>
  Not sure this counts as a derivative work, though.
  Because the entire tech stack is brand-new, the implementation is very different.

* The model weights I distribute were converted from `Mistral-7B-instruct-v0.1`, Apache-2.0 license.<br/>
  This is definitely a derivative work, I distribute the original tensors converted to another, compressed data format.

* Many compute shaders are from my [Whisper project](https://github.com/Const-me/Whisper/tree/master/ComputeShaders), MPL-2.0 license.<br/>
  Some parts of the C++ backend DLL are also [from there](https://github.com/Const-me/Whisper/tree/master/Whisper).

* The software depends on my [ComLightInterop](https://github.com/Const-me/ComLightInterop) library, MIT license.

* `TorchLoader` project includes copy-pasted pieces of [pickle library](https://github.com/irmen/pickle), MIT license, with substantial changes.

* `SentencePiece` static library is copy-pasted [from Google](https://github.com/google/sentencepiece), Apache-2.0 license.<br/>
  No changes there, but I have implemented COM API around the library.

* `MistralChat.FolderPicker` class was copy-pasted [from stackoverflow](https://stackoverflow.com/a/66187224/126995).