import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input  # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array  # type: ignore
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout  # type: ignore
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  # type: ignore
from sklearn.metrics import classification_report, confusion_matrix

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
CLASSES = ['Diseased', 'Healthy', 'Nutrient_Deficiency']

# Dictionary mapping classes to 1-line eco-friendly explanations
EXPLANATIONS = {
    'Healthy': 'Crop is healthy. Continue standard organic maintenance.',
    'Diseased': 'Pathogen detected. Isolate affected leaves and apply neem oil / bio-fungicide immediately.',
    'Nutrient_Deficiency': 'Nutrient imbalance detected. Verify soil N-P-K levels and apply organic compost or vermicompost.'
}

def create_data_generators(dataset_dir):
    """
    Sets up the ImageDataGenerator for training and validation splits.
    Applies MobileNetV2 specific preprocessing and data augmentation.
    """
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2 # 80/20 train/val split
    )

    train_generator = datagen.flow_from_directory(
        dataset_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    val_generator = datagen.flow_from_directory(
        dataset_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False # DO NOT shuffle validation data for accurate evaluation tracking
    )

    return train_generator, val_generator

def build_model(num_classes):
    """
    Builds the MobileNetV2 model using Transfer Learning.
    Adds a custom top layer for our specific classification.
    """
    # Load pretrained MobileNetV2, ignoring the top fully connected layers
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze the base layers so we don't accidentally ruin the pretrained weights
    base_model.trainable = False

    # Custom top layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(dataset_dir, epochs=10, model_save_path='models/crop_disease_model.h5'):
    """
    Complete pipeline to load data, build model, and train with callbacks.
    """
    print("Initializing Data Generators...")
    train_gen, val_gen = create_data_generators(dataset_dir)
    num_classes = train_gen.num_classes
    
    # Map model indices back to class names for later
    class_indices = train_gen.class_indices
    class_names = list(class_indices.keys())
    print(f"Classes Found: {class_names}")

    print("Building Model...")
    model = build_model(num_classes=num_classes)

    # Callbacks
    # Ensure directory exists for the model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True)

    print("Starting Training...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[early_stop, checkpoint]
    )

    print("Evaluating Model...")
    evaluate_model(model, val_gen, class_names)
    
    return model, class_names

def evaluate_model(model, val_gen, class_names):
    """
    Evaluates the model on the validation set, printing accuracy, 
    confusion matrix, and classification report.
    """
    # Reset generator to be sure it starts from the beginning
    val_gen.reset()
    
    # Evaluate
    loss, accuracy = model.evaluate(val_gen)
    print(f"\nFinal Validation Accuracy: {accuracy*100:.2f}%")
    print(f"Final Validation Loss: {loss:.4f}\n")

    # Predict val data
    print("Generating Classification Report and Confusion Matrix...")
    y_pred = model.predict(val_gen)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = val_gen.classes

    print("--- Classification Report ---")
    print(classification_report(y_true, y_pred_classes, target_names=class_names))

    print("--- Confusion Matrix ---")
    print(confusion_matrix(y_true, y_pred_classes))

def predict_disease(image_path, model_path='models/crop_disease_model.h5'):
    """
    Inference function for a single leaf image.
    Uses the trained lightweight model to predict class and return eco-friendly suggestions.
    """
    if not os.path.exists(model_path):
        return {"error": f"Model not found at {model_path}. Please train the model first."}
    
    model = tf.keras.models.load_model(model_path)
    
    # Load and preprocess image
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict
    predictions = model.predict(img_array)[0]
    
    # We must assume the classes order here matches our generator's alphabetical sorting
    # Default alphabetic: ['Diseased', 'Healthy', 'Nutrient_Deficiency']
    class_idx = np.argmax(predictions)
    # Re-verify the typical sorted order based on directory names
    class_name = sorted(CLASSES)[class_idx]
    confidence = float(predictions[class_idx]) * 100

    return {
        "class": class_name,
        "confidence": f"{confidence:.2f}%",
        "explanation": EXPLANATIONS.get(class_name, "No explanation available.")
    }

if __name__ == "__main__":
    # Test block to demonstrate usage
    test_dir = "data/plantdisease"
    
    # Setup mock data locally to prove it runs if not provided.
    import shutil
    from PIL import Image
    
    print("Checking for dataset. If missing, a tiny mock dataset will be created for testing.")
    if not os.path.exists(test_dir):
        print("Dataset not found. Creating mock dataset for 1 Epoch Test Drive...")
        for c in CLASSES:
            os.makedirs(os.path.join(test_dir, c), exist_ok=True)
            # Create a couple of fake 224x224 green square images representing 'leaves'
            for idx in range(15):  # Sufficient enough to pass 80/20 split validation
                img = Image.new('RGB', (224, 224), color = (34, 139, 34))
                img.save(os.path.join(test_dir, c, f'mock_{idx}.jpg'))
                
    print("\n--- Starting Pipeline ---")
    model_save_path = "models/crop_disease_model.h5"
    
    # Shorten epochs for the test run
    trained_model, class_list = train_model(
        dataset_dir=test_dir,
        epochs=1, 
        model_save_path=model_save_path
    )
    
    print("\n--- Testing Single Prediction Inference ---")
    test_image = os.path.join(test_dir, 'Healthy', 'mock_0.jpg')
    result = predict_disease(test_image, model_path=model_save_path)
    print(f"Prediction Output: {result}")
