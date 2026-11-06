import cv2
import mediapipe as mp
import numpy as np

#configuraci'on de mediapipe
mp_hands=mp.solutions.hands
hands=mp_hands.Hands(min_detection_confidence=0.5,min_tracking_confidence=0.5, max_num_hands=1)
mp_dibujo =mp.solutions.drawing_utils

cap=cv2.VideoCapture(0)

#para recordar la posición de la mano
xp,yp=0,0
lienzo=None

while True:
    exito,frame=cap.read()
    if not exito:
        print("No se pudo capturar la imagen")
        break

    #usar el efecto espejo de la camara para que concuerde con los movimientos
    frame=cv2.flip(frame,1)

    if lienzo is None:
        lienzo=np.zeros_like(frame)

    frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    resultados=hands.process(frame_rgb)

    #si detecta al menos una mano
    if resultados.multi_hand_landmarks:
        for hand_landmarks in resultados.multi_hand_landmarks:
            mp_dibujo.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            indice=hand_landmarks.landmark[8]
            medio=hand_landmarks.landmark[12]
            anular=hand_landmarks.landmark[16]
            menique=hand_landmarks.landmark[20]

            #los nudillos
            nudillo_indice=hand_landmarks.landmark[6]
            nudillo_medio=hand_landmarks.landmark[10]
            nudillo_anular=hand_landmarks.landmark[14]
            nudillo_menique=hand_landmarks.landmark[18]

            alto,ancho,_=frame.shape
            x=int(indice.x*ancho)
            y=int(indice.y*alto)

            mano_cerrada=(indice.y>nudillo_indice.y and medio.y>nudillo_medio.y and anular.y>nudillo_anular.y and menique.y>nudillo_menique.y)
            if mano_cerrada:
                lienzo=np.zeros_like(frame)
                cv2.putText(frame,"Borrando",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
                xp,yp=0,0
            else:
                if xp==0 and yp==0:
                    xp,yp=x,y
                #trayectoria de la mano
                cv2.line(lienzo,(xp,yp),(x,y), (255,0,255), 10)
                xp,yp=x,y
            #cursores de la mano
            cv2.circle(frame,(x,y),15,(0,0,255),cv2.FILLED)
    else:
        xp,yp=0,0

    lienzo_gray=cv2.cvtColor(lienzo,cv2.COLOR_BGR2GRAY)
    _,lienzo_inv=cv2.threshold(lienzo_gray,50,255,cv2.THRESH_BINARY_INV)
    lienzo_inv=cv2.cvtColor(lienzo_inv,cv2.COLOR_GRAY2BGR)
    frame=cv2.bitwise_and(frame,lienzo_inv)
    frame=cv2.bitwise_or(frame,lienzo)
    
    cv2.imshow("InvisiPen", frame)
    #si presionamos la tecla "q" se cierra la ventana
    if cv2.waitKey(1) & 0xFF==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()