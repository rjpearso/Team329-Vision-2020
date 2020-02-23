# CircuitPython demo - NeoPixel
import time
import board
import neopixel
 
pixel_pin = board.D5 # pwm pin, typically we use digital 5
num_pixels = 24
 
led = neopixel.NeoPixel(pixel_pin, num_pixels, brightness=1, auto_write=False) # brightness is in % 0.0-1.0 (0%-100%)

x = 0
green = [0, 255, 0]

while True:
  if(x<=24):
    led[x] = green 
    x+=1
    time.sleep(0.1)
 
