TEST_PRINT= True
def test_print(*args):
    """[同print函数] 只是在TEST_PRINT = False时不会输出。
    """    
    if TEST_PRINT:
        for arg in args:
            print(arg, end=' ')
    print()