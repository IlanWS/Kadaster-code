In this repo, the data is included, but in the name of replicability, the python scripts used to harvest these images
are provided and an explanation on how to run them correctly is given here. To collect the images in the folder "Data",
one needs to create a Web Mapping Service (WMS) layer in QGIS, configured with the MAP-files provided in the folder
"Map_config_files", by running the following command in WSL:

docker run -e MAPSERVER_CONFIG_FILE=/srv/data/example.conf -e MS_MAPFILE=/srv/data/brt-achtergrondkaart-standaard-alles-wit.map -e SERVICE_TYPE=WMS --rm -p 80:80 --name mapserver-example -v `pwd`/example:/srv/data pdok/mapserver

to get the input data map. To get the output data map, one runs the following:

docker run -e MAPSERVER_CONFIG_FILE=/srv/data/example.conf -e MS_MAPFILE=/srv/data/brt-achtergrondkaart-standaard-only-marked-labels.map -e SERVICE_TYPE=WMS --rm -p 80:80 --name mapserver-example -v `pwd`/example:/srv/data pdok/mapserver

Running the file "QGIS_code.py" in a python console in QGIS pans the viewing canvas such that HTTP requests can be
collected in a JSON file by opening the debugger/developer window (View > debugger/developer > requests), pressing
record, running the python code and waiting until finished. Once the code has finished running, stop recording and
download the requests log (note that there is a limit on how many HTTP requests will be saved in the file, meaning that
this processed ought to be repeated multiple times if >1000 image pairs need to be collected). The resulting JSON file
will resemble the JSON-file "zutphen_met_labels.json" as found in the folder "Data".

Running the file "JSON_download.py", while still connected to the WMS downloads the images, defined by the JSON file and
ran on localhost, to disk. Note that the JSON-file that is created with the 2 docker commands above would be identical,
making that to download the different images (input roadmap and output labels) one only needs to change the config of
the MAP-file that the WMS is run on. This makes that this script needs to be run twice. One time with the first docker
command, and the parameter "input_folder" (defined in file "config.py"), which will load the input data. A second time
with the second docker command, and the parameter "output folder" (also defined in "config.py"), to download the output
labels to the right directory (both in folder "Data").