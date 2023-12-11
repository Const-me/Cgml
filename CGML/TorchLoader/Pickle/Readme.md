Copy-pasted from there: https://github.com/irmen/pickle

Made substantual changes.

1. Most importantly, modified the `Unpickler.persistentLoad()` API.<br/>
The version in that project expects argument to be a single string.<br/>
The models I want to consume here instead supply an array of objects.
The original version passes `System.Object[]` string, so the payload is lost.<br/>
This was the main reason why I have copy-pasted the source codes, instead of using the nuget package.

2. Removed serialization support, we only need to de-serialize in this project.