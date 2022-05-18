

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("E:/6th Semester/ML Lab/Project/Model_keras_plantvillage_full")

CLASS_NAMES = ['Apple scab','Apple Black rot','Apple rust','Apple healthy','Blueberry healthy','Cherry Powdery mildew','Cherry healthy','Corn Cercospora Gray leaf spot','Corn Common rust','Corn Northern Leaf Blight','Corn healthy','Grape Black rot','Grape Esca','Grape Leaf blight','Grape healthy','Orange Haunglongbing','Peach Bacterial spot','Peach healthy','Pepper Bacterial spot','Pepper healthy','Potato Early blight','Potato Late blight','Potato healthy','Raspberry healthy','Soybean healthy','Squash Powdery mildew','Strawberry Leaf scorch','Strawberry healthy','Tomato Bacterial spot','Tomato Early blight','Tomato Late blight','Tomato Leaf Mold','Tomato Septoria leaf spot','Tomato Spider mites','Tomato Target Spot','Tomato Yellow Leaf Curl Virus','Tomato mosaic virus','Tomato healthy']
#CLASS_NAMES = ['Grape_Black_rot','Grape_Esca','Grape_Leaf_blight','Grape_healthy','Potato_Early_blight','Potato_Late_blight','Potato_healthy']
# CLASS_NAMES = ["","Potato_Early_blight","","","","",""]
#CLASS_NAMES = ["Strawberry scortch","Strawberry healthy"]


@app.get("/ping")
async def ping():
    return "Hello World"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image1 = read_file_as_image(await file.read())
    # image1.resize(224,224,3)
    
    print("KKK")
    print(type(image1))
    print(image1.shape)
    image3=Image.fromarray(image1)
    # image3.thumbnail((224,224))
    image = image3.resize((224,224))
    print("PPP")
    print(type(image))
    # image.show()

    # img_batch = np.expand_dims(image, 0)
    # print(type(img_batch))

    # preprocessed_image=tf.keras.applications.mobilenet.preprocess_input(image)
    # predictions = MODEL.predict(preprocessed_image)


    # image = tf.keras.preprocessing.image.load_img('E:/6th Semester/ML Lab/Project/potato-blight.jpg', target_size=(224, 224))
    # print("ZZZ")
    # print(type(image))
    # image.show()

    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])

    preprocessed_image=tf.keras.applications.mobilenet.preprocess_input(input_arr)
    predictions = MODEL.predict(preprocessed_image)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)



