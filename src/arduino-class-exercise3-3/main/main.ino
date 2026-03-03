int whichIncomingByte = 1;
int incomingByte1 = 0;
int incomingByte2 = 0;

void setup() {
  Serial.begin(9600);
}

void loop() {
  if (Serial.available() > 0) {
    if (whichIncomingByte == 1) {
      incomingByte1 = Serial.read();
      incomingByte1 = incomingByte1 - 48;
      whichIncomingByte = 2;
    }
    if (whichIncomingByte == 2) {
      incomingByte2 = Serial.read();
      incomingByte2 = incomingByte2 - 48;
      whichIncomingByte = 1;
      Serial.print("the sum is: ");
      Serial.println(incomingByte1 + incomingByte2, DEC);
    }
  }
}
