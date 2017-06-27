# Adding feature extraction algorithms

New feature extraction algorithms can be added to the crunher. All that is needed is needed is a function with the correct call signature and for the function to be added to the `fingerprint_dict` in crunch.py.

The call signature of feature extraction functions should be `f(numpy.ndarray, int, int) -> numpy.ndarray`. Where the input array is the waveform data, the first integer is sample rate and the secont integer is sample length. The return type should be a numpy.ndarray that allways has the same shape.

The new function should be made accessible to crunch.py and the function should be bound to the `fingerprint_dict` dictionary so that it can be called from the command line.

## Example: Adding `my_fingerprint`function to [`fingerprint.py`](../subprocesses/fingerprint.py)

Add a new function to `fingerprint.py`:

```python
def my_fingerprint(data: np.ndarray, sr: int, size: int) -> np.ndarray:
    """Generate fingerprint using my method
    ...
    """
    results = do_cool_stuff(data) # feature extraction goes here.
    return results
```

Add `my_fingerprint` to the subprocess [`__init__.py`](../subprocesses/__init__.py)

```python
__all__ = ["t_sne", ... , "my_fingerprint]
```
and

```python
from subprocesses.fingerprint import fft_fingerprint, .... , my_fingerprint
```

Add `my_fingerprint` to the `fingerprint_dict` in [`crunch.py`](../crunch.py)

```python
class ProcessFunctions:
    fingerprint_dict = {"fft": fft_fingerprint,
                        "chroma": chroma_fingerprint,
                        "ms": ms_fingerprint,
                        "mfcc": mfcc_fingerprint,
                        "my": my_fingerprint}

    ...
```

Now you can verify that everything works by seeing that `python3.6 crunch.py -h` lists `my` as a usable fingerprinting algorithm: `-g {fft,chroma,ms,mfcc,my}`. Now your algorithm should be called to do the fingerprinting when `-g my` is entered as a command line argument.
