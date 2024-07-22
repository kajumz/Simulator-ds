import hashlib
a = 1321
b = str(a).encode()
print(int(hashlib.sha256(b).hexdigest(), 16))