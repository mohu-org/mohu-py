import mohu as mu


print("=" * 60)
print("MOHU — RUST-POWERED ARRAYS FOR PYTHON")
print("=" * 60)

a = mu.array([[1.0, 2.0], [3.0, 4.0]])
b = mu.array([[5.0, 6.0], [7.0, 8.0]])

print()
print("1. ARRAY CREATION")
print(a)

print()
print("2. ARRAY METADATA")
print("shape:", a.shape)
print("ndim:", a.ndim)
print("dtype:", a.dtype)
print("strides:", a.strides)
print("size:", a.size)
print("values:", a.tolist())

print()
print("3. LAYOUT TRANSFORMATIONS")
print("transpose:")
print(a.T)
print("reshape:")
print(a.reshape((4, 1)))

print()
print("4. MATRIX MULTIPLICATION")
product = a @ b
print(product)
assert product.tolist() == [[19.0, 22.0], [43.0, 50.0]]

print()
print("=" * 60)
print("PYTHON DEMO COMPLETE")
print("=" * 60)
