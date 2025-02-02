import cv2
import os
from google.cloud import vision

# --- Configuration ---
# Google Cloud setup
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"C:\Users\seohj\Downloads\gen-lang-client-0591642241-1946d8bcbfab.json"  # Replace with your credentials path

# Initialize the Vision API client
client = vision.ImageAnnotatorClient()

# Proximity configuration
CLOSE_THRESHOLD = 0.5  # Normalized value for how close
REALLY_CLOSE_THRESHOLD = 0.75  # Normalized value for how REALLY close
OUTPUT_MESSAGE_CLOSE = "Person is too close!"
OUTPUT_MESSAGE_REALLY_CLOSE = "Person is REALLY close! Contacting authorities..."

output_triggered_close = False  # To avoid spamming 'close' message
output_triggered_really_close = False # To avoid spamming 'really close' message


def detect_people(frame):
    """Detects people in a frame using Google Cloud Vision API."""
    _, buffer = cv2.imencode('.jpg', frame)  # Encode frame to JPG
    image_content = buffer.tobytes()

    image = vision.Image(content=image_content)
    objects = client.object_localization(image=image).localized_object_annotations

    people_detected = []
    for obj in objects:
        if obj.name.lower() == "person":
            people_detected.append(obj)

    return people_detected


def calculate_proximity(frame, person_data):
    """Calculates the relative proximity of a person to the camera.
    Proximity is based on the relative width of the bounding box compared to the frame width.
    """
    h, w, _ = frame.shape
    vertices = person_data.bounding_poly.normalized_vertices
    x1 = vertices[0].x
    x2 = vertices[2].x
    person_width_normalized = x2 - x1
    return person_width_normalized


def draw_bounding_boxes(frame, people, close_people=[], really_close_people=[]):
    """Draws bounding boxes around detected people in the frame.
    Highlights bounding boxes with different colors based on proximity.
    """
    for person in people:
        vertices = person.bounding_poly.normalized_vertices
        h, w, _ = frame.shape
        x1 = int(vertices[0].x * w)
        y1 = int(vertices[0].y * h)
        x2 = int(vertices[2].x * w)
        y2 = int(vertices[2].y * h)

        if person in really_close_people:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red if really close
        elif person in close_people:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)  # Orange if close
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green otherwise
    return frame


def main():
    global output_triggered_close, output_triggered_really_close
    cap = cv2.VideoCapture(0)  # Open default camera (0) or provide the camera index if necessary

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()  # Capture frame
        if not ret:
            print("Error: Can't receive frame.")
            break

        people_detections = detect_people(frame)
        close_people = []
        really_close_people = []

        # Analyze people proximity
        for person in people_detections:
            proximity = calculate_proximity(frame, person)

            if proximity >= REALLY_CLOSE_THRESHOLD:
                really_close_people.append(person)
                if not output_triggered_really_close:
                    print(OUTPUT_MESSAGE_REALLY_CLOSE)
                    output_triggered_really_close = True  # Prevent repeated really close messages
            elif proximity >= CLOSE_THRESHOLD:
                close_people.append(person)
                if not output_triggered_close:
                    print(OUTPUT_MESSAGE_CLOSE)
                    output_triggered_close = True # Prevent repeated close messages

            elif output_triggered_close and proximity < CLOSE_THRESHOLD:
                output_triggered_close = False  # Allow to trigger again
            elif output_triggered_really_close and proximity < REALLY_CLOSE_THRESHOLD:
                output_triggered_really_close = False # Allow to trigger again


        frame_with_boxes = draw_bounding_boxes(frame, people_detections, close_people, really_close_people)
        cv2.imshow("Person Detection", frame_with_boxes)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()