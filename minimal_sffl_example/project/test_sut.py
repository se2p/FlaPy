from sut import gen_int


# flaky test
def test_1():
    assert "Small" in gen_int()


# stable test
def test_2():
    assert "number" in gen_int()
