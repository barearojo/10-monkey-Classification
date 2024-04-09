import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QDesktopWidget
from PyQt5.QtGui import QPixmap

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
        self.btn_cargar.setGeometry(150, 10, 100, 30)
        self.btn_cargar.clicked.connect(self.cargar_imagen)

        self.btn_ejecutar = QPushButton("Ejecutar Preddición", self)
        self.btn_ejecutar.setGeometry(1000, 800, 150, 30)
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
        filename, _ = QFileDialog.getOpenFileName(self, "Abrir Archivo", "", "Archivos de Imagen (*.png *.jpg *.bmp)")
        if filename:
            self.imagen = QPixmap(filename)
            self.label_imagen.setPixmap(self.imagen)

    def ejecutar_funcion(self):
        if self.imagen:
            # Llamar a tu función existente pasándole la imagen
            resultado = tu_funcion_existente(self.imagen)
            # Hacer algo con el resultado, como mostrarlo en la interfaz
            print(resultado)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

