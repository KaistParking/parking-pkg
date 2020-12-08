#include <stdio.h>
#include <Servo.h>

#define SERVO_PIN 10

Servo servo;


int E1 = 5;     //M1 Speed Control
int M1 = 4;     //M1 Direction Control
float global_speed = 0.;
float deg2control_ratio = 2.16;
//float deg2control_ratio = 1.8;
float delay_time = 0.6;

float max_speed_car = 4.0; // 4
float speed2pwm_ratio = 255. / max_speed_car;
float max_speed = 0.65;

void stop_car(void) {
  digitalWrite(E1, 0);
  digitalWrite(M1, LOW);
}
void forward(char a) {
  analogWrite(E1, 100);
  digitalWrite(M1, HIGH);
  delay(50);
  analogWrite(E1, a);      //PWM Speed Control
  digitalWrite(M1, HIGH);
}
void backward(char a) {
  analogWrite(E1, 100);
  digitalWrite(M1, LOW);
  delay(50);
  analogWrite(E1, a);
  digitalWrite(M1, LOW);
}


void setup() {
  // setup
  Serial.begin(9600);      //Set Baud Rate
  for(int i = 4; i <= 5; i++)
    pinMode(i, OUTPUT);
  servo.attach(SERVO_PIN);
  digitalWrite(E1, LOW);
  servo.write(95);

  Serial.println("Run control... q10");
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
      float angle = abs_steer;
      if (angle > 30) angle = 30;
      if (handle[0] == 'L') angle = 95 + angle*deg2control_ratio;
      else if (handle[0] == 'R') angle = 95 - angle*deg2control_ratio;
      else angle = 95;
      // control
      servo.write(int(angle));

      //// 2) change speed
      // parse
//      float speed2pwm_ratio = 25;
//      float pwm = abs_accel;
//      if (pwm > 0) pwm = 45;
//      else if (pwm == 0) pwm = 0;
//      // control
//      if (gear[0] == 'F') forward(50);
//      else if (gear[0] == 'B') backward(50);
//      else stop_car();

      if (gear[0] == 'F') global_speed += (abs_accel*delay_time);
      else if (gear[0] == 'B') global_speed -= (abs_accel*delay_time);
      if (global_speed < -max_speed) global_speed = -max_speed;
      else if (global_speed > max_speed) global_speed = max_speed;
      
      int pwm = abs(global_speed*speed2pwm_ratio);
      if (gear[0] == 'F') forward(pwm);
      else if (gear[0] == 'B') backward(pwm*1.2);
      else stop_car();
  
      delay(int(delay_time*1000));
    }
  }
  else {
    stop_car();
  }

}
