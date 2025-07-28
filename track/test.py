import cv2


def check_cameras():
    for index in range(0, 11):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # CAP_DSHOW helps on Windows
        if not cap.isOpened():
            print(f"No camera at index {index}")
            cap.release()
            continue

        print(f"\nTrying camera index: {index}")
        print("Press any key to check next camera or 'q' to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"Couldn't read frame from index {index}")
                break

            cv2.imshow(f"Camera Index: {index}", frame)

            # Wait for key press (1ms) and check if window was closed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

            # Check if window was closed manually
            if cv2.getWindowProperty(f"Camera Index: {index}", cv2.WND_PROP_VISIBLE) < 1:
                break

        cap.release()
        cv2.destroyAllWindows()

    print("\nFinished checking all camera indexes (0-10)")
    print("If you didn't see any camera feed:")
    print("1. Check your camera connection")
    print("2. Try different USB ports")
    print("3. Verify camera works in other applications")


# Run the camera checker
check_cameras()