import glob as gb
invar=gb.iglob("*.py")
print(type(invar))
print("list of files")
for py in invar:
    print(py)