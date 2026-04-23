import cv2
import numpy as np

# Change this to whichever video you want to calibrate
VIDEO_PATH = "../model2/dataset/TrackNetV2_Dataset/Professional/match3/video/2_18_15.mp4"

def get_calibration_points():
    print("===============================================")
    print(" COURT CALIBRATION TOOL")
    print("===============================================")
    print("Which lines are you clicking to calibrate?")
    print("1: Singles Court (Inner sidelines)")
    print("2: Doubles Court (Outer sidelines)")
    choice = input("Enter 1 or 2: ").strip()
    
    # Court coords are (x, y) in metres from centre.
    # x: -6.7 = near baseline, +6.7 = far baseline
    # y: -2.59/-3.05 = right sideline, +2.59/+3.05 = left sideline
    # Click order matches screen corners: TL → TR → BR → BL
    # TL pixel = far-left corner  → (+6.7, +2.59)
    # TR pixel = far-right corner → (+6.7, -2.59)
    # BR pixel = near-right corner → (-6.7, -2.59)
    # BL pixel = near-left corner  → (-6.7, +2.59)
    if choice == '1':
        court_pts = [[6.7, 2.59], [6.7, -2.59], [-6.7, -2.59], [-6.7, 2.59]]
        target_name = "SINGLES"
    else:
        court_pts = [[6.7, 3.05], [6.7, -3.05], [-6.7, -3.05], [-6.7, 3.05]]
        target_name = "DOUBLES"

    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not read video at {VIDEO_PATH}")
        return
        
    points = []
    display_frame = frame.copy()
    
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:
                points.append([x, y])
                cv2.circle(display_frame, (x, y), 5, (0, 0, 255), -1)
                
                if len(points) > 1:
                    cv2.line(display_frame, tuple(points[-2]), tuple(points[-1]), (0, 255, 0), 2)
                if len(points) == 4:
                    cv2.line(display_frame, tuple(points[-1]), tuple(points[0]), (0, 255, 0), 2)
                    
                cv2.imshow('Calibration', display_frame)
                
                if len(points) == 4:
                    print("\nAll 4 points selected! Press any key to close the window.")

    print(f"\nPlease click the 4 corners of the {target_name} court")
    print("in the following EXACT order:")
    print("1. Top-Left corner")
    print("2. Top-Right corner")
    print("3. Bottom-Right corner")
    print("4. Bottom-Left corner")
    print("\n(Go clockwise starting from the top-left)")
    
    cv2.imshow('Calibration', display_frame)
    cv2.setMouseCallback('Calibration', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap.release()
    
    if len(points) == 4:
        print("\nSUCCESS! Here is the code snippet for your scoring_main.py:")
        print("---------------------------------------------------------")
        print("engine.calibrate(")
        print(f"    pixel_pts={points},")
        print(f"    court_pts={court_pts},")
        print(")")
        print("---------------------------------------------------------")
    else:
        print("\nCalibration cancelled.")

if __name__ == "__main__":
    get_calibration_points()
