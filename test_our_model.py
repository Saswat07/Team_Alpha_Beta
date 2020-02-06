import cv2
import sys
from model.darkflow.net.build import TFNet
import argparse
if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--threshold",help="Threshold")
    parser.add_argument("--image",help="image path")

    args=parser.parse_args()
    threshold=0.05
    if args.threshold is not None:
        threshold=args.threshold
    if args.image is None:
        print("Enter image path")
        sys.exit(0)


    options={
    'model':'model/cfg/our_model.cfg',
    'load':1,                
    'threshold':threshold,
    'gpu':1.0
    }
    tfnet=TFNet(options)

    img=cv2.imread(args.image)
    result=tfnet.return_predict(img)
    print(result)
    for i in result:
            tl = (i['topleft']['x'],i['topleft']['y'])
            br = (i['bottomright']['x'],i['bottomright']['y'])
            label = i['label']
            
            cv2.rectangle(img, tl, br, (0, 255, 0), 2)
            img = cv2.putText(img, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        #img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (1280,720))
    cv2.imshow('output',img)
    cv2.waitKey()
    