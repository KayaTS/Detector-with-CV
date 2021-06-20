from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(rotation_range=60,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=False,
                             vertical_flip=False,
                             fill_mode='nearest')

imgcount = 140
for a in range(imgcount):
    img = load_img('b (%d).jpg' % (a+2))
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    a+=1
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir='../octocopter-artirma', 
                              save_format='jpg'):
        i += 1
        if i > 5:
            break 

