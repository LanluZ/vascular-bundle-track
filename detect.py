import cv2

from ultralytics import YOLO


def main():
    model = YOLO("best.pt")

    results = model('test.jpg')

    annotated_img = results[0].plot(line_width=3, font_size=16)

    cv2.imwrite('annotated_img.jpg', annotated_img)


if __name__ == '__main__':
    main()
