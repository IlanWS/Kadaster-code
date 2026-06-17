# Run this script in QGIS to harvest images information into a json file. Not to be ran outsinde of QGIS.
# Resulting JSON file is e.g. enschede_LR.json, containing a bbox and link for the local host. These results are used in JSON_download

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
    newx= x+(i+i)*100
    for j in range(25):
        loop = QEventLoop()
        newy= y - (j+1)*100
        point = QgsPointXY(newx, newy)
        canvas.setCenter(point)
        canvas.mapCanvasRefreshed.connect(loop.quit)
        canvas.refresh()
        loop.exec_()
        time.sleep(0.1)


