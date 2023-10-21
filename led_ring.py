from pixel_ring import apa102
from pixel_ring import pixel_ring
from gpiozero import LED

NUM_LED = 12
BRIGHTNESS = 10
ORDER = "rbg"

class LEDRing:
    def __init__(self):
        self.strip = apa102.APA102(num_led=NUM_LED, global_brightness=BRIGHTNESS, order=ORDER)
    
    def light_led(self, angle):
        led_idx = (round(angle/30)-2)%12
        
        self.strip.set_pixel_rgb(led_idx, 0xFF0000, bright_percent = 1) # Red
        
        self.strip.show()
        
        return led_idx
        
    def clear(self):
        self.strip.clear_strip()
        