def test_self_polluter():
    # assert that file does NOT exists and create it if not
    self_polluter_file = Path("/tmp/.minimal_flaky_example_self_polluter_file")
    file_existed = self_polluter_file.exists()
    self_polluter_file.touch(exist_ok=True)
    assert not file_existed

def test_self_statesetter():
    # assert that file exists and create it if not
    self_state_setter_file = Path("/tmp/.minimal_flaky_example_self_state_setter_file")
    file_existed = self_state_setter_file.exists()
    self_state_setter_file.touch(exist_ok=True)
    assert file_existed
