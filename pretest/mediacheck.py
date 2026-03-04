import mediapipe as mp
import importlib, importlib.util, pkgutil

print('mp.__version__ =', getattr(mp, '__version__', None))
print('top-level attrs =', [a for a in dir(mp) if not a.startswith('_')])
print('find_spec("mediapipe.solutions") =', importlib.util.find_spec('mediapipe.solutions'))
print('find_spec("mediapipe.tasks.python.vision") =', importlib.util.find_spec('mediapipe.tasks.python.vision'))

print('\niterating mediapipe package contents (pkgutil.iter_modules):')
for mod in pkgutil.iter_modules(mp.__path__):
    print(' -', mod.name)

try:
    from mediapipe import solutions
    print('\nimport mediapipe.solutions OK, attrs =', [a for a in dir(solutions) if not a.startswith('_')])
except Exception as e:
    print('\nimport mediapipe.solutions error:', type(e).__name__, e)

try:
    import mediapipe.tasks.python.vision as t
    print('\nimport mediapipe.tasks.python.vision OK')
except Exception as e:
    print('\nimport mediapipe.tasks.python.vision error:', type(e).__name__, e)
