TEST_PRINT= True
def test_print(*args):
    if TEST_PRINT:
        for arg in args:
            print(arg, end=' ')
    print()