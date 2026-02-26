#run this script in QGIS to harvest images into a json file
import time

#huidige extent van de GUI
canvas = iface.mapCanvas()

# Startcoordinaten(linkerbovenhoek)in EPSG:28992 (Amersfoort)
x = 210000
y = 462000

#startpunt
point = QgsPointXY(x, y)
canvas.setCenter(point)
canvas.zoomScale(2500)
canvas.refresh()

#loop over area
for i in range(25):
    newx= x+i*100
    for j in range(25):
        loop = QEventLoop()
        newy= y - j*100
        point = QgsPointXY(newx, newy)
        canvas.setCenter(point)
        canvas.mapCanvasRefreshed.connect(loop.quit)
        canvas.refresh()
        loop.exec_()
        time.sleep(0.1)

#canvas.zoomByFactor(1.5)
