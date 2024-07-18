# cf. https://github.com/pypa/manylinux/issues/53

GOOD_SSL = "https://google.com"
BAD_SSL = "https://self-signed.badssl.com"

import sys


from urllib.parse import urlparse

def is_valid_scheme(url):
    parsed_url = urlparse(url)
    return parsed_url.scheme in ('http', 'https')

print("Testing SSL certificate checking for Python:", sys.version)

if (sys.version_info[:2] < (2, 7)
    or sys.version_info[:2] < (3, 4)):
    print("This version never checks SSL certs; skipping tests")
    sys.exit(0)

if sys.version_info[0] >= 3:
    from urllib.request import urlopen
    EXC = OSError
else:
    from urllib import urlopen
    EXC = IOError

print("Connecting to %s should work" % (GOOD_SSL,))
if not is_valid_scheme(GOOD_SSL):
    raise ValueError("Invalid URL scheme")
urlopen(GOOD_SSL)
print("...it did, yay.")

from urllib.parse import urlparse

def is_valid_scheme(url):
    parsed_url = urlparse(url)
    return parsed_url.scheme in ('http', 'https')

print("Connecting to %s should fail" % (BAD_SSL,))
try:
    if not is_valid_scheme(BAD_SSL):
        raise ValueError("Invalid URL scheme")
    urlopen(BAD_SSL)
    # If we get here then we failed:
    print("...it DIDN'T!!!!!11!!1one!")
    sys.exit(1)
except (EXC, ValueError):
    print("...it did, yay.")
