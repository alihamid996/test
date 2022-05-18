
name='raw'
MODEL_PATH = '30_04_2022_10_21_48.h5'
model = load_model(MODEL_PATH)
def model_predict(img):

    
    prediction = model.predict(img,batch_size=1,steps=1)

    d = prediction.flatten()

    return list(d).index(d.max())+  1

imagee="6d51bd53-2556-410c-b151-547c83a06287___RS_LB 3276_final_masked"
path="C:/Users/ASUS/Desktop/shaima/data_raw/raw/test/23/{}.jpg".format(imagee)
shutil.copy(path, "C:/Users/ASUS/Desktop/test/1/{}.jpg".format(imagee))


test_gen=tf.keras.preprocessing.image.ImageDataGenerator(

    rescale=1/255.,

)
image_width=256
image_width=256

test_data=test_gen.flow_from_directory(
 "C:/Users/ASUS/Desktop/test/",
    target_size=(256, 256),
    color_mode='rgb',
    classes=None,
    class_mode=None,
    batch_size=1,
    shuffle=False,
)

ro=model_predict(test_data)
os.remove("C:/Users/ASUS/Desktop/test/1/{}.jpg".format(imagee))#
