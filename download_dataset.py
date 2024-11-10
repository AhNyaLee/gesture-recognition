from roboflow import Roboflow
rf = Roboflow(api_key="DwvX5W1wO0PutXu9jcJ6")
project = rf.workspace("ann-itrkh").project("slovo-hvkjg")
version = project.version(4)
dataset = version.download("yolov8")
