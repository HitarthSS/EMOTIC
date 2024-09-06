import cv2
import numpy as np
import torch
from torchvision import transforms

cap = cv2.VideoCapture(1)  # Assuming camera index 1
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduce width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Reduce height


def infer_emotions(image_context, bbox, models, context_norm, body_norm, face_norm, ind2cat, ind2vad):
    ''' Infer emotions using the Emotic model for a detected person in the image. '''
    height, width, _ = image_context.shape
    x1, y1, x2, y2 = bbox

    face_y1 = int(y1)
    face_y2 = int(y1 + (y2 - y1) * 0.5)

    x1, face_y1, x2, face_y2 = [max(0, min(coord, max_dim)) for coord, max_dim in zip([x1, face_y1, x2, face_y2], [width, height, width, height])]

    body_image = image_context[y1:y2, x1:x2]
    face_image = image_context[face_y1:face_y2, x1:x2]

    if body_image.size == 0 or face_image.size == 0:
        print("Warning: Empty body or face image after cropping, skipping this detection.")
        return [], [0.0, 0.0, 0.0]

    body_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(body_norm[0], body_norm[1])
    ])

    face_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(face_norm[0], face_norm[1])
    ])

    context_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(context_norm[0], context_norm[1])
    ])

    body_image = body_transform(cv2.cvtColor(body_image, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
    face_image = face_transform(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
    context_image = context_transform(cv2.cvtColor(image_context, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)

    with torch.no_grad():
        body_features = models[1](body_image)
        face_features = models[2](face_image)
        context_features = models[0](context_image)
        cat_output, cont_output = models[3](context_features, body_features, face_features)

    cat_probs = torch.sigmoid(cat_output[0])
    top_probs, top_indices = torch.topk(cat_probs, 5)  # Get the top 5 emotions
    selected_emotions = [ind2cat[idx.item()] for idx in top_indices[2:]]  # Select 3rd to 5th emotions

    pred_cont = cont_output[0].cpu().numpy()

    return selected_emotions, pred_cont

# Ensure the directory containing emotic.py is in the Python path
import sys
sys.path.append('/Users/hitarthshukla/Desktop/SPIStest')

# Import the Emotic class
from emotic import Emotic

# Load YOLO
net = cv2.dnn.readNet("/Users/hitarthshukla/Desktop/SPIStest/yolov3 (1).weights", "/Users/hitarthshukla/Desktop/SPIStest/yolov3 (1).cfg")
layer_names = net.getLayerNames()

# Fix for different OpenCV versionsq
layer_indexes = net.getUnconnectedOutLayers()
if layer_indexes.ndim == 1:  # Scalar values directly indicate the indices
    output_layers = [layer_names[index - 1] for index in layer_indexes]
else:  # The indices are wrapped inside an array
    output_layers = [layer_names[index[0] - 1] for index in layer_indexes]

# Load COCO labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize webcam
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not open video stream or file")
    exit()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Emotic models including the face model
model_context = torch.load("/Users/hitarthshukla/Downloads/emotic model v2/models/model_context1.pth").to(device)
model_body = torch.load("/Users/hitarthshukla/Downloads/emotic model v2/models/model_body1.pth").to(device)
model_face = torch.load("/Users/hitarthshukla/Downloads/emotic model v2/models/model_face1.pth").to(device)
emotic_model = torch.load("/Users/hitarthshukla/Downloads/emotic model v2/models/model_emotic1.pth").to(device)
models = [model_context, model_body, model_face, emotic_model]

# Set models to evaluation mode
model_context.eval()
model_body.eval()
model_face.eval()
emotic_model.eval()

# Continue the rest of your code as before...

# Emotion categories and normalization values
cat = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval', 'Disconnection', 
       'Disquietment', 'Doubt/Confusion', 'Embarrassment', 'Engagement', 'Esteem', 'Excitement', 'Fatigue', 'Fear', 'Happiness', 
       'Pain', 'Peace', 'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning']
cat2ind = {emotion: idx for idx, emotion in enumerate(cat)}
ind2cat = {idx: emotion for idx, emotion in enumerate(cat)}

vad = ['Valence', 'Arousal', 'Dominance']
ind2vad = {idx: continuous for idx, continuous in enumerate(vad)}

context_norm = [0.4690646, 0.4407227, 0.40508908]
context_std = [0.2514227, 0.24312855, 0.24266963]
body_norm = [0.43832874, 0.3964344, 0.3706214]
body_std = [0.24784276, 0.23621225, 0.2323653]

# Assuming these are your normalization values for the face model
face_mean = [0.4046, 0.4416, 0.4685]  # Example values, adjust based on your model
face_std = [0.2431, 0.2441, 0.2521]   # Example values, adjust based on your model
face_norm = [face_mean, face_std]



  # Assuming the inference function is in emotic_infer.py

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    height, width, channels = frame.shape  # Define 'width' and 'height' here

    # Prepare the image for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Analyze detection results
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.55 and classes[class_id] == "person":
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, x + w, y + h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-max suppression to eliminate redundant overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and infer emotions
    # In the main loop where detections are processed
    for i in range(len(boxes)):
        if i in indexes:
            x, y, x2, y2 = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Call infer_emotions with all required parameters
            pred_cat, pred_cont = infer_emotions(frame, [x, y, x2, y2], models, context_norm, body_norm, face_norm, ind2cat, ind2vad)

            
            # Display predicted emotions and VAD scores
            for idx, emotion in enumerate(pred_cat):
                cv2.putText(frame, emotion, (x, y + (idx + 1) * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            vad_text = 'VAD: %.2f, %.2f, %.2f' % (pred_cont[0], pred_cont[1], pred_cont[2])
            cv2.putText(frame, vad_text, (x, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # Press Q on keyboard to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
