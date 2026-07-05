from ultralytics import YOLO

model = YOLO("yolo26n.pt")  # initialize model
results = model("path/to/image.jpg")  # perform inference
results[0].show()  # display results for the first image