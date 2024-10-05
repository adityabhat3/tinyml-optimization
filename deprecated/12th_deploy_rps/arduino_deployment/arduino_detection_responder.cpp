/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if defined(ARDUINO) && !defined(ARDUINO_ARDUINO_NANO33BLE)
#define ARDUINO_EXCLUDE_CODE
#endif  // defined(ARDUINO) && !defined(ARDUINO_ARDUINO_NANO33BLE)

#ifndef ARDUINO_EXCLUDE_CODE

#include "detection_responder.h"

#include "Arduino.h"

// Flash the blue LED after each inference
void RespondToDetection(tflite::ErrorReporter* error_reporter,
                        int8_t rock_score, int8_t paper_score , int8_t scissors_score) {
  static bool is_initialized = false;
  if (!is_initialized) {
    // Pins for the built-in RGB LEDs on the Arduino Nano 33 BLE Sense
    pinMode(LEDR, OUTPUT); //rock
    pinMode(LEDG, OUTPUT); //paper
    pinMode(LEDB, OUTPUT); //scissor
    is_initialized = true;
  }

  // Note: The RGB LEDs on the Arduino Nano 33 BLE
  // Sense are on when the pin is LOW, off when HIGH.

  // Switch the person/not person LEDs off
  digitalWrite(LEDR, HIGH);
  digitalWrite(LEDG, HIGH);
  digitalWrite(LEDB, HIGH);
  delay(10000);

  if (rock_score > paper_score) {
    if(rock_score > scissors_score){
      digitalWrite(LEDR, LOW);
    }
    else{
      digitalWrite(LEDB, LOW);
    }
  } 
  else {
    if(paper_score > scissors_score){
      digitalWrite(LEDG, LOW);
    }
    else{
      digitalWrite(LEDB, LOW);
    }
  }

  int rock_percentage = (int)rock_score;
  int paper_percentage = (int)paper_score;
  int scissors_percentage = (int)scissors_score;

  TF_LITE_REPORT_ERROR(error_reporter, "Rock percentage: %d Paper percentage: %d Scissors percentage: %d",
                      rock_percentage, paper_percentage, scissors_percentage);
}

#endif  // ARDUINO_EXCLUDE_CODE
