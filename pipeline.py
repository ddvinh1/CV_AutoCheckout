import time
from pathlib import Path
from classification_utils import warmup_dr, warmup_fd, classify_image
import subprocess
import os
import tensorflow as tf
from datetime import datetime,date

def get_files_in_folder(folder_path):
    folder = Path(folder_path)
    return {file.name: file.stat().st_mtime for file in folder.glob('*') if file.is_file()}

def count_files_in_folder(folder_path):
    folder = Path(folder_path)
    files = [file for file in folder.glob('*') if file.is_file()]
    return len(files)

def load_model_for_folder(folder_path):
    if "food" in folder_path:
        return food_model, class_food
    elif "drink" in folder_path:
        return drink_model, class_drink
    else:
        pass

def predict_all_files(selected_model, labels, folder_path):
    files = [file for file in Path(folder_path).iterdir() if file.is_file()]
    for file in files:
        classify_image(selected_model, labels, file)
def detect_changes(folder_paths):
    previous_states = {folder_path: get_files_in_folder(folder_path) for folder_path in folder_paths}

    try:
        while True:
            time.sleep(1.5)

            for folder_path in folder_paths:
                current_state = get_files_in_folder(folder_path)
                added_files = set(current_state) - set(previous_states[folder_path])
                deleted_files = set(previous_states[folder_path]) - set(current_state)
                modified_files = {file for file in set(current_state) if file in previous_states[folder_path] and current_state[file] > previous_states[folder_path][file]}

                if added_files or modified_files:
                    os.system('cls')
                    now = datetime.now()
                    now_d = date.today()
                    current_time = now.strftime("%H:%M:%S")
                    current_date = now_d.strftime("%d/%m/%Y")
                    print("\nCurrent Time =", current_date, current_time)
                    print("Your bill include: \n")
                    # Use the pre-loaded model based on the detected folder
                    selected_model, labels = load_model_for_folder(folder_path)

                    predict_all_files(selected_model, labels, folder_path)

                previous_states[folder_path] = current_state


    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    warm_fd_pth = r"img\1.jpg"
    with open('food_labels.txt', 'r', encoding='utf-8') as file:
        class_food = [line.strip() for line in file.readlines()]
    food_model = tf.keras.models.load_model('food.keras')


    warm_dr_pth = r"img\2.jpg"
    with open('drink_labels.txt', 'r', encoding='utf-8') as file:
        class_drink = [line.strip() for line in file.readlines()]
    drink_model = tf.keras.models.load_model('drink.keras')


    folder_paths = [r"yolov5\runs\detect\exp\crops\drink",
                    r"yolov5\runs\detect\exp\crops\food"]

    warmup_fd(food_model, class_food, warm_fd_pth)
    warmup_dr(drink_model, class_drink, warm_dr_pth)

    subprocess.run("yolo.bat")

    detect_changes(folder_paths)




















































