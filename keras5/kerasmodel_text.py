
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import cv2

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)


model = tensorflow.keras.models.load_model('keras_model.h5')


data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

testi = 0
testix = 1
for t in range(150):
    img = Image.open('../archive/program/test/all/b (%d).jpg' % (testix)) 
    image = img.convert('RGB')
    output = cv2.imread('../archive/program/test/all/b (%d).jpg'% (testix))

    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    prediction = model.predict(data)

    if(100*prediction[0,0] > 100*prediction[0,1]):
        #print("%", 100*prediction[0,0] , " ihtimalle drone")
        text = 'drone %.2f' % (100*prediction[0,0])
        color = (0, 255, 0)
        cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    color, 2, cv2.LINE_AA)
        cv2.imwrite("../archive/program/testresults/results/13-keras/pos(%d).jpg" % (testix), output)
    else:
        #print("%", 100*prediction[0,1] , " ihtimalle drone değil")
        text = 'not drone %.2f' % (100*prediction[0,1])
        color = (0, 0, 255)
        cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        color, 2, cv2.LINE_AA)
        cv2.imwrite("../archive/program/testresults/results/13-keras/neg(%d).jpg" % (testix), output)
    testix = testix+1
