import cv2
import mediapipe as mp
import numpy as np
import torch
import string
from modelo import LetrasCNN

#clases y la carga del modelo entrenado
alafabeto= list(string.ascii_uppercase)
modelo=LetrasCNN(num_clases=26)
modelo.load_state_dict(torch.load("modelo_letras.pth", map_location=torch.device('cpu')))
modelo.eval()

#configuraci'on de mediapipe
mp_hands=mp.solutions.hands
hands=mp_hands.Hands(min_detection_confidence=0.5,min_tracking_confidence=0.5, max_num_hands=1)
mp_dibujo =mp.solutions.drawing_utils

cap=cv2.VideoCapture(0)

#para recordar la posición de la mano
xp,yp=0,0
lienzo=None
palabra_guardada=""
letra_actual=""

while True:
    exito,frame=cap.read()
    if not exito:
        print("No se pudo capturar la imagen")
        break

    #usar el efecto espejo de la camara para que concuerde con los movimientos
    frame=cv2.flip(frame,1)
    alto,ancho,_=frame.shape

    if lienzo is None:
        lienzo=np.zeros_like(frame)

    frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    resultados=hands.process(frame_rgb)

    #si detecta al menos una mano
    if resultados.multi_hand_landmarks:
        for hand_landmarks in resultados.multi_hand_landmarks:
            mp_dibujo.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            pulgar=hand_landmarks.landmark[4]
            indice=hand_landmarks.landmark[8]
            medio=hand_landmarks.landmark[12]
            anular=hand_landmarks.landmark[16]
            menique=hand_landmarks.landmark[20]

            #los nudillos
            nudillo_pulgar=hand_landmarks.landmark[2]
            nudillo_indice=hand_landmarks.landmark[6]
            nudillo_medio=hand_landmarks.landmark[10]
            nudillo_anular=hand_landmarks.landmark[14]
            nudillo_menique=hand_landmarks.landmark[18]

            x=int(indice.x*ancho)
            y=int(indice.y*alto)

            pulgar_arriba= pulgar.y<nudillo_pulgar.y
            indice_arriba=indice.y<nudillo_indice.y
            medio_arriba=medio.y<nudillo_medio.y
            anular_arriba=anular.y<nudillo_anular.y
            menique_arriba=menique.y<nudillo_menique.y

            #MODO GUARDAR con la mano abierta, 4 dedos levantados
            if indice_arriba and medio_arriba and anular_arriba and menique_arriba:
                cv2.putText(frame,"Guardando",(20,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
                if letra_actual:
                    palabra_guardada+=letra_actual
                    letra_actual=""
                    lienzo=np.zeros_like(frame)#autoborrado
                    cv2.putText(frame,f"Palabra guardada: {palabra_guardada}",(20,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
                xp,yp=0,0
            #MODO BORRADOR con el puño cerrado
            elif indice_arriba and not medio_arriba and not anular_arriba and menique_arriba:
                cv2.putText(frame,"Borrando",(20,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
                lienzo=np.zeros_like(frame)
                letra_actual=""
                xp,yp=0,0
            #MODO MOVIMIENTO con una V
            elif indice_arriba and medio_arriba and not anular_arriba and not menique_arriba:
                cv2.putText(frame,"Moviendo",(20,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3)
                cv2.rectangle(frame, (x-15,y-15), (x+15,y+15), (255,0,0), cv2.FILLED)
                xp,yp=0,0
            #MODO DIBUJO con el dedo índice levantado y el medio abajo
            elif indice_arriba and not medio_arriba and not anular_arriba and not menique_arriba:
                cv2.putText(frame,"Dibujando",(20,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),3)
                cv2.circle(frame,(x,y),15,(255,0,255),cv2.FILLED)
                if xp==0 and yp==0:
                    xp,yp=x,y
                cv2.line(lienzo,(xp,yp),(x,y),(255,0,255),10)
                xp,yp=x,y

            #MODO ESPACIO con el pulgar y meñique levantados(los demás no)
            elif pulgar_arriba and not indice_arriba and not medio_arriba and not anular_arriba and menique_arriba:
                cv2.putText(frame, "Espacio", (20,50), cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,255),3)
                if len(palabra_guardada)> 0 and palabra_guardada[-1] != " ":
                    palabra_guardada += " "

                lienzo=np.zeros_like(frame)
                letra_actual=""
                xp,yp =0,0
    else:
        xp,yp=0,0

    lienzo_gray=cv2.cvtColor(lienzo,cv2.COLOR_BGR2GRAY)
    _,lienzo_bin=cv2.threshold(lienzo_gray,10,255,cv2.THRESH_BINARY)
    contornos,_=cv2.findContours(lienzo_bin,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
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
        roi=lienzo_bin[y1:y2,x1:x2]

        if roi.size>0:
            #preprocesamiento de la ROI para el modelo
            h_roi,w_roi=roi.shape
            lado_maximo=max(h_roi,w_roi)
            pad_top=(lado_maximo-h_roi)//2
            pad_bottom=lado_maximo-h_roi-pad_top
            pad_left=(lado_maximo-w_roi)//2
            pad_right=lado_maximo-w_roi-pad_left

            roi_cuadrado=cv2.copyMakeBorder(roi,pad_top,pad_bottom,pad_left,pad_right,cv2.BORDER_CONSTANT,value=0)
            roi_redimensionado=cv2.resize(roi_cuadrado,(28,28), interpolation=cv2.INTER_AREA)
            cv2.imshow("ROI redimensionado",roi_redimensionado)
            roi_normalizado=roi_redimensionado.astype(np.float32)/255.0
            roi_normalizado=(roi_normalizado-0.5)/0.5
            tensor_entrada=torch.tensor(roi_normalizado).unsqueeze(0).unsqueeze(0)  #agregar dimensiones de batch y canal
            #inferencia con el modelo en tiempo real
            with torch.no_grad():
                predicciones=modelo(tensor_entrada)
                probabilidades=torch.nn.functional.softmax(predicciones,dim=1)
                confianza, clase_predicha=torch.max(probabilidades,1)
                letra_actual=alafabeto[clase_predicha.item()]
                porcentaje_confianza=confianza.item()*100
                cv2.putText(frame,f"Letra: {letra_actual} ({porcentaje_confianza:.2f}%)",(x_min,y_min-10),cv2.FONT_HERSHEY_COMPLEX,1.5,(255,105,180),3)
    else:
        #si no hay contornos, se cierra la ventana de ROI
        letra_actual=""
        try:
            cv2.destroyWindow("ROI redimensionado")   
        except:
            pass

    cv2.putText(frame,f"Palabra guardada: {palabra_guardada}_",(50,alto-100),cv2.FONT_HERSHEY_COMPLEX,1.5,(0,255,255),4)

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