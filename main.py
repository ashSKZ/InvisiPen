import cv2

cap=cv2.VideoCapture(0)
while True:
    exito,frame=cap.read()
    if not exito:
        print("No se pudo capturar la imagen")
        break

    #usar el efecto espejo de la camara para que concuerde con los movimientos
    frame=cv2.flip(frame,1)
    cv2.imshow("Pizarra", frame)
    #si presionamos la tecla "q" se cierra la ventana
    if cv2.waitKey(1) & 0xFF==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()