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

            indice_arriba=indice.y<nudillo_indice.y
            medio_arriba=medio.y<nudillo_medio.y
            anular_arriba=anular.y<nudillo_anular.y
            menique_arriba=menique.y<nudillo_menique.y

            #MODO BORRADOR con el puño cerrado
            if not indice_arriba and not medio_arriba and not anular_arriba and not menique_arriba:
                lienzo=np.zeros_like(frame)
                cv2.putText(frame,"Borrando",(20,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
                xp,yp=0,0
            #MODO MOVIMIENTO con una V
            elif indice_arriba and medio_arriba and not anular_arriba and not menique_arriba:
                cv2.putText(frame,"Moviendo",(20,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3)
                cv2.rectangle(frame, (x-15,y-15), (x+15,y+15), (255,0,0), cv2.FILLED)
                xp,yp=0,0
            #MODO DIBUJO con el dedo índice levantado y el medio abajo
            elif indice_arriba and not medio_arriba:
                cv2.putText(frame,"Dibujando",(20,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),3)
                cv2.circle(frame,(x,y),15,(255,0,255),cv2.FILLED)
                if xp==0 and yp==0:
                    xp,yp=x,y
                cv2.line(lienzo,(xp,yp),(x,y),(255,0,255),10)
                xp,yp=x,y
    else:
        xp,yp=0,0

    lienzo_gray=cv2.cvtColor(lienzo,cv2.COLOR_BGR2GRAY)

    contornos,_=cv2.findContours(lienzo_gray,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if contornos:
        alto,ancho,_=frame.shape
        x_min,y_min=ancho,alto
        x_max,y_max=0,0
        for c in contornos:
            x_c,y_c,w_c,h_c=cv2.boundingRect(c)
            if x_c< x_min: x_min =x_c
            if y_c< y_min: y_min =y_c
            if x_c+w_c> x_max: x_max =x_c+w_c
            if y_c+h_c> y_max: y_max =y_c+h_c

        cv2.rectangle(frame, (x_min,y_min),(x_max,y_max),(0,255,0),2)
        #ROIs con padding
        padding=20
        y1=max(0,y_min-padding)
        y2=min(alto,y_max+padding)
        x1=max(0,x_min-padding)
        x2=min(ancho,x_max+padding)
        roi=lienzo[y1:y2,x1:x2]

        if roi.size>0:
            cv2.imshow("ROI",roi)
    else:
        #si no hay contornos, se cierra la ventana de ROI
        try:
            cv2.destroyWindow("ROI")   
        except:
            pass


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