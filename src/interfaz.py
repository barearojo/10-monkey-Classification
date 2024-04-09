import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Interfaz con Imagen")
        self.setGeometry(100, 100, 400, 300)

        self.label_imagen = QLabel(self)
        self.label_imagen.setGeometry(50, 50, 300, 200)
        self.label_imagen.setPixmap(QPixmap(""))

        self.btn_cargar = QPushButton("Cargar Imagen", self)
        self.btn_cargar.setGeometry(150, 10, 100, 30)
        self.btn_cargar.clicked.connect(self.cargar_imagen)

        self.btn_ejecutar = QPushButton("Ejecutar Función", self)
        self.btn_ejecutar.setGeometry(150, 260, 100, 30)
        self.btn_ejecutar.clicked.connect(self.ejecutar_funcion)

        self.imagen = None

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

