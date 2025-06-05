# Create labels.txt file with class labels

# List of class labels
class_labels = ['Apple Scab Leaf', 'Apple leaf', 'Apple rust leaf', 'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Bell_pepper leaf', 'Bell_pepper leaf spot', 'Blueberry leaf', 'Blueberry___healthy', 'Cherry leaf', 'Cherry_(including_sour)__Powdery_mildew', 'Cherry(including_sour)__healthy', 'Corn Gray leaf spot', 'Corn leaf blight', 'Corn rust leaf', 'Corn(maize)__Cercospora_leaf_spot Gray_leaf_spot', 'Corn(maize)_Common_rust', 'Corn(maize)__Northern_Leaf_Blight', 'Corn(maize)healthy', 'Grape___Black_rot', 'Grape___Esca(Black_Measles)', 'Grape___Leaf_blight(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing(Citrus_greening)', 'Peach leaf', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato leaf early blight', 'Potato leaf late blight', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry leaf', 'Raspberry___healthy', 'Soyabean leaf', 'Soybean___healthy', 'Squash Powdery mildew leaf', 'Squash___Powdery_mildew', 'Strawberry leaf', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato Early blight leaf', 'Tomato Septoria leaf spot', 'Tomato leaf', 'Tomato leaf bacterial spot', 'Tomato leaf late blight', 'Tomato leaf mosaic virus', 'Tomato leaf yellow virus', 'Tomato mold leaf', 'Tomato two spotted spider mites leaf', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy', 'grape leaf', 'grape leaf black rot']

# Write labels to file
with open('PestDetectionProject/models/labels.txt', 'w') as file:
    for label in sorted(class_labels):
        file.write(label + '\n')

print(f"✅ Created labels.txt with {len(class_labels)} class labels")
print("Each label is written on a separate line.")