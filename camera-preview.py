import cv2
import argparse
import sys
from datetime import datetime


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Simple Camera Preview Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview default camera
  python camera_preview.py

  # Preview specific camera device
  python camera_preview.py --device 2

  # Display with specific window size
  python camera_preview.py --width 1280 --height 720

  # Show FPS counter
  python camera_preview.py --show-fps
        """
    )

    parser.add_argument(
        '--device',
        type=int,
        default=0,
        help='Camera device index (default: 0)'
    )
    parser.add_argument(
        '--width',
        type=int,
        help='Display width (default: camera native resolution)'
    )
    parser.add_argument(
        '--height',
        type=int,
        help='Display height (default: camera native resolution)'
    )
    parser.add_argument(
        '--show-fps',
        action='store_true',
        help='Show FPS counter'
    )
    parser.add_argument(
        '--list-devices',
        action='store_true',
        help='List available camera devices and exit'
    )

    return parser.parse_args()


def list_camera_devices():
    """List available camera devices"""
    available_devices = []
    for i in range(10):  # Check first 10 indexes
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                available_devices.append((i, width, height, fps))
            cap.release()
    return available_devices


def calculate_fps(start_time, frame_count):
    """Calculate current FPS"""
    elapsed_time = (datetime.now() - start_time).total_seconds()
    return frame_count / elapsed_time if elapsed_time > 0 else 0


def main():
    """Main function"""
    args = parse_arguments()

    # List devices if requested
    if args.list_devices:
        print("\nAvailable camera devices:")
        devices = list_camera_devices()
        if not devices:
            print("No cameras found")
            return
        for device in devices:
            print(f"Device {device[0]}: {device[1]}x{device[2]} @ {device[3]}fps")
        return

    # Open camera
    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        print(f"Error: Could not open camera device {args.device}")
        sys.exit(1)

    # Set custom resolution if specified
    if args.width and args.height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # Get actual camera resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\nCamera Preview")
    print(f"Device: {args.device}")
    print(f"Resolution: {width}x{height}")
    print(f"Press 'q' to quit, 's' to save screenshot\n")

    # FPS calculation variables
    if args.show_fps:
        start_time = datetime.now()
        frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            # Add FPS counter
            if args.show_fps:
                frame_count += 1
                fps = calculate_fps(start_time, frame_count)
                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

            # Show frame
            cv2.imshow('Camera Preview', frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('s'):
                # Save screenshot
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"camera_snapshot_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")

    except KeyboardInterrupt:
        print("\nProgram terminated by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
