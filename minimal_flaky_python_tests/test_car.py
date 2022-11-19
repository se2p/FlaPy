class Car:
    blinker = 'OFF'  # OFF / LEFT / RIGHT / BOTH
    hazard_lights_on = False

    def set_turn_signal(self, direction):
        if not self.hazard_lights_on:
            self.blinker = direction

    def toggle_hazard_lights(self):
        self.hazard_lights_on = not self.hazard_lights_on
        self.blinker = 'BOTH'


car = Car()


def test_turn_signal():
    car.set_turn_signal('LEFT')
    assert car.blinker == 'LEFT'


def test_hazard_lights():
    car.toggle_hazard_lights()
    assert car.blinker == 'BOTH'
