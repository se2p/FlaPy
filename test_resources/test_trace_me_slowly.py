import time


class SlowList(list):
    def __init__(self, waiting_time):
        self.waiting_time = waiting_time
        super().__init__()

    def __iter__(self):
        time.sleep(self.waiting_time)
        return super().__iter__()


# This function is needed, because the return-value of __init__ is None
def generate_slow_list(waiting_time):
    return SlowList(waiting_time)


def test_expensive_hash():
    sl = generate_slow_list(2)
    sl.append("foo")
