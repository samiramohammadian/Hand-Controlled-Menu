#import datetime 
#strt = datetime.datetime.now()

# imports 
import cv2
import mediapipe as mp
import time
import subprocess
import numpy as np 


# Initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
            max_num_hands=1, 
            min_detection_confidence=0.5, 
            model_complexity = 0 )  


#mp_drawing = mp.solutions.drawing_utils

# Configuration
WIDTH, HEIGHT = 640, 480
RELATION = WIDTH/1280
DIAMETER = int(RELATION * 150)
HOVER_DURATION = 1
font_scale = RELATION

# Formal color values
BUTTON_COLOR = (48,25,52)    # Dark Purple 
HOVER_COLOR = (177,156,217)  # Light Purple 

# Dictionary for button 
MENU_BUTTONS = [
    {"name": "  play game", 
    "top_left": (WIDTH // 4 - DIAMETER, HEIGHT // 2 - DIAMETER),
    "bottom_right": (WIDTH // 4 + DIAMETER, HEIGHT // 2 + DIAMETER), 
    "hover_start": None, 
    "color": BUTTON_COLOR},

    {"name": "    exit",
    "top_left": (3 * WIDTH // 4 - DIAMETER, HEIGHT // 2 - DIAMETER),
    "bottom_right": (3 * WIDTH // 4 + DIAMETER, HEIGHT // 2 + DIAMETER),
    "hover_start": None, 
    "color": BUTTON_COLOR}
]

 
# checking that x,y "landmark8" and x , y rectangle 
def is_point_inside_rectangle(point, rectangle):
    
    x, y = point
    top_left_x, top_left_y = rectangle["top_left"]
    bottom_right_x, bottom_right_y = rectangle["bottom_right"]
    return top_left_x <= x <= bottom_right_x and top_left_y <= y <= bottom_right_y


def show_start_page():
    """
    Displays a simple color background for the start page.
    """
    # Create a simple colored frame with the desired dimensions
    start_frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    # Set the background color
    background_color = (48, 25, 52)  # Dark Purple (RGB: 48, 25, 52)

    # Fill the frame with the background color
    start_frame[:, :] = background_color

    # Add "Welcome" text
    start_text = "Welcome!"
    start_text_size = cv2.getTextSize(start_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 2, 2)[0]
    start_text_x = WIDTH // 2 - start_text_size[0] // 2
    start_text_y = HEIGHT // 2 + start_text_size[1] // 2
    cv2.putText(start_frame, start_text, (start_text_x, start_text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 2,
                (255, 255, 255), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Hand-Tracking Menu', start_frame)
    cv2.waitKey(1000)


def handle_selection(choice):
    if choice == "  play game":
        subprocess.run(["python", "simpleProject.py"])
        print("Starting game...")
    elif choice == "    exit":
        exit()



def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 15)
    show_start_page()
    
    frame_count = 0 
    while cap.isOpened():
        success, image = cap.read()
        
        if not success:
            print("Ignoring empty camera frame.")
            continue
          
        
        # Flip the image horizontally for a mirror-like effect
        frame_count += 1
        if frame_count % 2 != 0 :
                
            image = cv2.flip(image, 1)

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = hands.process(image_rgb)
            #print(results.multi_hand_landmarks)

            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0] # first hand that detect 
                index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_x, index_y = int(index_tip.x * image.shape[1]), int(index_tip.y * image.shape[0]) # normalization, to become pixel 

                for button in MENU_BUTTONS:
                    inside = is_point_inside_rectangle((index_x, index_y), button)
                    color = button["color"]
                    
                    if inside:
                        if "hover_start" in button and button["hover_start"] is not None:
                            elapsed_time = time.time() - button["hover_start"]
                        else:
                            elapsed_time = time.time()

                        alpha = min(1, elapsed_time / HOVER_DURATION)

                        hover_red = int(alpha * HOVER_COLOR[0] + (1 - alpha) * BUTTON_COLOR[0])
                        hover_green = int(alpha * HOVER_COLOR[1] + (1 - alpha) * BUTTON_COLOR[1])
                        hover_blue = int(alpha * HOVER_COLOR[2] + (1 - alpha) * BUTTON_COLOR[2])
                        color = (hover_red, hover_green, hover_blue)



                    # Draw the button as a rectangle
                    top_left = button["top_left"]
                    bottom_right = button["bottom_right"]
                    cv2.rectangle(image, top_left, bottom_right, color, -1)  # -1 fills the rectangle

                    # Draw the text with shadow
                    text_size = cv2.getTextSize(button["name"], cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
                    text_x = top_left[0] + (bottom_right[0] - top_left[0]) // 2 - text_size[0] // 2
                    text_y = top_left[1] + (bottom_right[1] - top_left[1]) // 2 + text_size[1] // 4 
                    cv2.putText(image, button["name"], (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale , (255, 255, 255), 1, cv2.LINE_AA)  # Starlight white

                    # Update hover_start time
                    if inside and not button["hover_start"]:
                        button["hover_start"] = time.time()
                    elif not inside:
                        button["hover_start"] = None

                    # Trigger the action if hovered for more than the specified duration
                    if inside and button["hover_start"] and (time.time() - button["hover_start"] > HOVER_DURATION):
                        handle_selection(button["name"])
                        button["hover_start"] = None

            cv2.imshow('Hand-Tracking Menu', image)

            # endt = datetime.datetime.now()
            # runt = endt - strt
            # print(runt.microseconds)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    hands.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
