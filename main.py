import cv2
import mediapipe as mp

import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_dibujo

#configuraci'on de mediapipe
hands=mp_hands.Hands(min_detection_confidence=0.5,min_tracking_confidence=0.5, max_num_hands=1)


cap=cv2.VideoCapture(0)
print("¡Cámara encendida! Entrando al ciclo de video...")
while True:
    exito,frame=cap.read()
    if not exito:
        print("No se pudo capturar la imagen")
        break

    #usar el efecto espejo de la camara para que concuerde con los movimientos
    frame=cv2.flip(frame,1)

    frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    resultados=hands.process(frame_rgb)

    #si detecta al menos una mano
    if resultados.multi_hand_landmarks:
        print("✅ ¡MANO DETECTADA!")
        for hand_landmarks in resultados.multi_hand_landmarks:
            mp_dibujo.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            indice=hand_landmarks.landmark[8]
            alto,ancho,_=frame.shape
            x=int(indice.x*ancho)
            y=int(indice.y*alto)

            cv2.circle(frame,(x,y),15,(0,0,255),cv2.FILLED)
    else:
        # Si MediaPipe no ve nada, imprimirá esto sin parar
        print("❌ Buscando mano...")

    cv2.imshow("Pizarra", frame)
    #si presionamos la tecla "q" se cierra la ventana
    if cv2.waitKey(1) & 0xFF==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()