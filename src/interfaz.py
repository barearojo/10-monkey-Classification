import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QDesktopWidget
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QLabel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Interfaz con Imagen")
        self.setGeometry(0, 0, 1500, 1500) #Primer argumento posicion inical horizontal,Pimer argumento posicion inicial, ancho, altura
        self.center()

        self.label_imagen = QLabel(self)
        self.label_imagen.setGeometry(50, 50, 800, 800)
        self.label_imagen.setPixmap(QPixmap(""))

        self.btn_cargar = QPushButton("Cargar Imagen", self)
        self.btn_cargar.setGeometry(150, 10, 250, 80)
        self.btn_cargar.clicked.connect(self.cargar_imagen)

        self.btn_ejecutar = QPushButton("Ejecutar Preddición", self)
        self.btn_ejecutar.setGeometry(850, 800, 250, 100)
        self.btn_ejecutar.clicked.connect(self.ejecutar_funcion)

        self.imagen = None

    def center(self):
        # Obtener la geometría de la ventana y del escritorio
        window_geometry = self.frameGeometry()
        desktop_center = QDesktopWidget().availableGeometry().center()
        # Centrar la ventana en el escritorio
        window_geometry.moveCenter(desktop_center)
        self.move(window_geometry.topLeft())

    def cargar_imagen(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Abrir Archivo", "", "Archivos de Imagen (*.png *.jpg *.bmp)") # es para indicar que no se usara el segundo valor devuelto por la funcion
        if filename:
            self.imagen = QPixmap(filename)
            self.label_imagen.setPixmap(self.imagen)


    def ejecutar_funcion(self):
        if self.imagen:
            # Crear un QLabel para mostrar el texto
            self.label_resultado = QLabel("Hola Mundo", self)
            # Agregar el QLabel a la ventana principal o a un layout según sea necesario
            # Por ejemplo, si self es la ventana principal:
            self.label_resultado.move(50, 50)  # Mover el QLabel a la posición deseada
            self.label_resultado.show()  # Mostrar el QLabel


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

