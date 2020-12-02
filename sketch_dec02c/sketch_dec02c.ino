#include <stdio.h>
#include <Servo.h>

#define SERVO_PIN       5
#define MOTOR_SPEED_PIN 6
#define MOTOR_DI_PIN    7

Servo servo;

void setup() {
  // setup
  Serial.begin(9600);
  Serial.println("start!");
  pinMode(MOTOR_DI_PIN, OUTPUT);
  pinMode(MOTOR_SPEED_PIN, OUTPUT);
  servo.attach(SERVO_PIN);
  
  // initialize
  analogWrite(MOTOR_DI_PIN, HIGH);
  analogWrite(MOTOR_SPEED_PIN, 0);
  servo.write(95);
}

void loop() {
  if (Serial.available()) {
    String sig = "";
    char check[2], handle[2], gear[2];
    char steer_buf[5], accel_buf[5];
    float abs_steer=0, abs_accel=0;

    // read input
    while (Serial.available()) {
      char wait = Serial.read();
      sig.concat(wait);
      delay(2);
    }
    sig.substring(0, 1).toCharArray(check, 2);
    Serial.print("input: ");
    Serial.println(sig);

//    // check
//    if (check[0] == 'Q') {
//      Serial.print("arrived: ");
//      Serial.print(sig);
//      Serial.print("length: ");
//      Serial.println(sig.length());
//    }

    // parse & control
    if (check[0] == 'Q' && sig.length() >= 12) {
      sig.substring(1, 2).toCharArray(handle, 2);
      sig.substring(2, 3).toCharArray(gear, 2);
      sig.substring(3, 7).toCharArray(steer_buf, 5);
      sig.substring(8, 12).toCharArray(accel_buf, 5);
      
      abs_steer = atof(steer_buf);
      abs_accel = atof(accel_buf);
  
      Serial.print("Steer: "); Serial.print(handle[0]); Serial.println(abs_steer);
      Serial.print("Accel: "); Serial.print(gear[0]); Serial.println(abs_accel);

      //// 1) change steer
      // parse
      float deg2control_ratio = 2.16;
      float angle = abs_steer;
      if (angle > 30) angle = 30;
//      else if (angle > 25) angle = 25;
//      else if (angle > 20) angle = 20;
//      else if (angle > 15) angle = 15;
//      else if (angle > 10) angle = 10;
//      else if (angle > 5) angle = 5;
//      else angle = 0;
      if (handle[0] == 'L') angle = 95 - angle*deg2control_ratio;
      else if (handle[0] == 'R') angle = 95 + angle*deg2control_ratio;
      else angle = 95;
      // control
      servo.write(int(angle));

      //// 2) change speed
      // parse
      float speed2pwm_ratio = 25;
      float pwm = abs_accel;
      if (pwm > 0) pwm = 45;
      else if (pwm == 0) pwm = 0;
      // control
      if (gear[0] == 'F') digitalWrite(MOTOR_DI_PIN, HIGH);
      else if (gear[0] == 'B') digitalWrite(MOTOR_DI_PIN, LOW);
      else digitalWrite(MOTOR_DI_PIN, HIGH);
      analogWrite(MOTOR_SPEED_PIN, int(pwm));
  
      delay(1000);
    }
  }
  else {
    analogWrite(MOTOR_SPEED_PIN, 0);
  }

}
