from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

import io
import os

import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms



# APP


app = FastAPI(
    title="CropGuard AI Service",
    description="AI service for crop disease detection",
    version="2.0.0"
)



# CORS


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# DEVICE


device = torch.device("cpu")

print("Using device:", device)



# MODEL PATH


MODEL_PATH = os.path.join(
    "models",
    "cropguard_resnet18.pth"
)


# LOAD MODEL


print("Loading CropGuard model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

class_names = checkpoint["class_names"]

print("Classes:", class_names)


model = models.resnet18(
    weights=None
)

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    len(class_names)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()

print("✅ CropGuard model loaded successfully!")


# IMAGE TRANSFORM


transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])



# DISEASE INFORMATION


# DISEASE INFORMATION

DISEASE_INFO = {

    # =========================
    # TOMATO
    # =========================

    "Tomato_Healthy": {
        "status": "Healthy Crop",
        "riskLevel": "Low",
        "treatment": [
            "No disease treatment is required",
            "Continue balanced irrigation and crop care",
            "Monitor plants regularly for early symptoms"
        ],
        "prevention": [
            "Maintain proper spacing between plants",
            "Keep the field free from weeds and plant debris",
            "Avoid unnecessary leaf wetness"
        ]
    },

    "Tomato_Early_Blight": {
        "status": "Disease Detected",
        "riskLevel": "High",
        "treatment": [
            "Remove and safely dispose of severely infected leaves",
            "Use a locally recommended fungicide according to the label",
            "Avoid excessive irrigation and prolonged leaf wetness"
        ],
        "prevention": [
            "Maintain good air circulation between plants",
            "Avoid overhead irrigation when possible",
            "Remove infected crop debris after harvest"
        ]
    },

    "Tomato_Late_Blight": {
        "status": "Disease Detected",
        "riskLevel": "High",
        "treatment": [
            "Remove severely infected plant material",
            "Use a locally recommended late-blight fungicide according to the label",
            "Improve field drainage and reduce prolonged leaf wetness"
        ],
        "prevention": [
            "Avoid overhead irrigation",
            "Maintain good air circulation",
            "Inspect plants frequently during cool and humid weather"
        ]
    },


    # =========================
    # POTATO
    # =========================

    "Potato_Healthy": {
        "status": "Healthy Crop",
        "riskLevel": "Low",
        "treatment": [
            "No disease treatment is required",
            "Continue proper irrigation and nutrient management",
            "Monitor leaves regularly"
        ],
        "prevention": [
            "Use healthy planting material",
            "Maintain proper field drainage",
            "Remove diseased plant debris from the field"
        ]
    },

    "Potato_Early_Blight": {
        "status": "Disease Detected",
        "riskLevel": "High",
        "treatment": [
            "Remove severely affected leaves",
            "Use a recommended fungicide according to the product label",
            "Maintain balanced irrigation and avoid plant stress"
        ],
        "prevention": [
            "Practice crop rotation where possible",
            "Avoid prolonged leaf wetness",
            "Remove infected crop residues"
        ]
    },

    "Potato_Late_Blight": {
        "status": "Disease Detected",
        "riskLevel": "High",
        "treatment": [
            "Remove severely infected plant material",
            "Apply a locally recommended late-blight fungicide as directed",
            "Avoid irrigation conditions that keep foliage wet for long periods"
        ],
        "prevention": [
            "Use healthy seed tubers",
            "Maintain good field drainage",
            "Monitor crops closely during cool and humid conditions"
        ]
    },


    # =========================
    # RICE
    # =========================

    "Rice___Healthy": {
        "status": "Healthy Crop",
        "riskLevel": "Low",
        "treatment": [
            "No disease treatment is required",
            "Continue proper water and nutrient management",
            "Monitor leaves and panicles regularly"
        ],
        "prevention": [
            "Use healthy and suitable seed",
            "Maintain proper field sanitation",
            "Avoid excessive nitrogen application"
        ]
    },

    "Rice___Brown_Spot": {
        "status": "Disease Detected",
        "riskLevel": "Moderate",
        "treatment": [
            "Maintain balanced crop nutrition",
            "Improve soil fertility where nutrient deficiency is suspected",
            "Use a recommended fungicide if disease severity is high"
        ],
        "prevention": [
            "Use healthy quality seed",
            "Maintain adequate potassium and other nutrients",
            "Avoid prolonged crop stress and poor field management"
        ]
    },

    "Rice___Leaf_Blast": {
        "status": "Disease Detected",
        "riskLevel": "High",
        "treatment": [
            "Remove or manage severely affected plant material",
            "Use a locally recommended blast-control fungicide when required",
            "Avoid excessive nitrogen fertilization"
        ],
        "prevention": [
            "Use resistant or recommended varieties where available",
            "Maintain balanced fertilizer application",
            "Monitor the crop closely during humid conditions"
        ]
    },

    "Rice___Neck_Blast": {
        "status": "Disease Detected",
        "riskLevel": "High",
        "treatment": [
            "Use a recommended fungicide at the appropriate crop stage",
            "Follow local agricultural guidance for blast management",
            "Maintain balanced crop nutrition"
        ],
        "prevention": [
            "Use healthy and recommended seed varieties",
            "Avoid excessive nitrogen application",
            "Monitor the crop around flowering and panicle development"
        ]
    },


    # =========================
    # WHEAT
    # =========================

    "wheat_healthy": {
        "status": "Healthy Crop",
        "riskLevel": "Low",
        "treatment": [
            "No disease treatment is required",
            "Continue balanced irrigation and nutrition",
            "Regularly inspect leaves and stems"
        ],
        "prevention": [
            "Use certified healthy seed",
            "Maintain field sanitation",
            "Follow suitable crop rotation practices"
        ]
    },

    "wheat_Crown and Root Rot": {
        "status": "Disease Detected",
        "riskLevel": "High",
        "treatment": [
            "Remove severely affected plants where practical",
            "Improve soil and drainage management",
            "Use locally recommended seed or soil treatment when appropriate"
        ],
        "prevention": [
            "Use healthy treated seed",
            "Maintain good soil drainage",
            "Practice crop rotation to reduce disease pressure"
        ]
    },

    "wheat_Leaf Rust": {
        "status": "Disease Detected",
        "riskLevel": "High",
        "treatment": [
            "Monitor rust development across the field",
            "Use a recommended fungicide when disease reaches the locally advised threshold",
            "Follow the fungicide label for application and safety"
        ],
        "prevention": [
            "Use rust-resistant varieties where available",
            "Use healthy quality seed",
            "Regularly inspect leaves for new rust symptoms"
        ]
    },

    "wheat_Loose Smut": {
        "status": "Disease Detected",
        "riskLevel": "High",
        "treatment": [
            "Remove and safely dispose of affected plants where practical",
            "Use certified seed for the next crop",
            "Use an appropriate seed treatment recommended locally"
        ],
        "prevention": [
            "Use certified disease-free seed",
            "Use recommended fungicidal seed treatment",
            "Avoid saving seed from infected fields"
        ]
    },


    # =========================
    # CHILLI
    # =========================

    "Chilli__healthy": {
        "status": "Healthy Crop",
        "riskLevel": "Low",
        "treatment": [
            "No disease treatment is required",
            "Continue balanced irrigation and nutrition",
            "Monitor leaves and fruits regularly"
        ],
        "prevention": [
            "Maintain proper spacing between plants",
            "Keep the field clean and weed-free",
            "Inspect plants regularly for pests and diseases"
        ]
    },

    "Chilli__Anthracnos": {
        "status": "Disease Detected",
        "riskLevel": "High",
        "treatment": [
            "Remove severely infected fruits and plant parts",
            "Use a locally recommended fungicide according to the label",
            "Avoid excessive moisture around the crop"
        ],
        "prevention": [
            "Use healthy disease-free planting material",
            "Improve air circulation between plants",
            "Remove infected fruits and crop debris"
        ]
    },

    "Chilli__Damping_Off": {
        "status": "Disease Detected",
        "riskLevel": "High",
        "treatment": [
            "Remove severely affected seedlings",
            "Reduce excessive watering and improve drainage",
            "Use locally recommended nursery treatment when necessary"
        ],
        "prevention": [
            "Use clean nursery soil and containers",
            "Avoid overwatering",
            "Provide proper drainage and ventilation"
        ]
    },

    "Chilli__Leaf_Curl_Virus": {
        "status": "Disease Detected",
        "riskLevel": "High",
        "treatment": [
            "Remove severely infected plants where practical",
            "Control insect vectors such as whiteflies using locally recommended methods",
            "Avoid using planting material from infected plants"
        ],
        "prevention": [
            "Use healthy planting material",
            "Monitor and manage whitefly populations",
            "Remove infected plants and nearby weed hosts"
        ]
    },

    "Chilli__Leaf_Spot": {
        "status": "Disease Detected",
        "riskLevel": "Moderate",
        "treatment": [
            "Remove severely affected leaves",
            "Use a locally recommended fungicide or bactericide after confirming the cause",
            "Avoid prolonged leaf wetness"
        ],
        "prevention": [
            "Maintain proper plant spacing",
            "Avoid overhead irrigation",
            "Remove infected leaves and crop debris"
        ]
    },

    "Chilli__Veinal_Mottle_Virus": {
        "status": "Disease Detected",
        "riskLevel": "High",
        "treatment": [
            "Remove severely infected plants where practical",
            "Manage insect vectors using locally recommended methods",
            "Do not use infected plants as planting material"
        ],
        "prevention": [
            "Use healthy planting material",
            "Control insect vectors and weeds",
            "Regularly inspect young leaves for symptoms"
        ]
    },

    "Chilli__Whitefly": {
        "status": "Pest Detected",
        "riskLevel": "High",
        "treatment": [
            "Monitor the underside of leaves for whiteflies",
            "Use suitable integrated pest management methods",
            "If necessary, use a locally registered insecticide according to its label"
        ],
        "prevention": [
            "Remove weeds that can host whiteflies",
            "Use yellow sticky traps for monitoring",
            "Regularly inspect the underside of leaves"
        ]
    },

    "Chilli__Yellowish": {
        "status": "Possible Crop Stress",
        "riskLevel": "Moderate",
        "treatment": [
            "Check the plant for nutrient deficiency, water stress or disease symptoms",
            "Maintain balanced irrigation",
            "Confirm the cause before applying pesticides or fertilizers"
        ],
        "prevention": [
            "Maintain balanced plant nutrition",
            "Avoid both overwatering and severe water stress",
            "Monitor leaves regularly for changes"
        ]
    }
}


# HOME


@app.get("/")
def home():

    return {

        "message": "CropGuard AI Service is running!",

        "status": "success",

        "model": "ResNet18",

        "classes": class_names
    }



# HEALTH CHECK


@app.get("/health")
def health_check():

    return {

        "status": "healthy",

        "model_loaded": True
    }


# PREDICTION


@app.post("/predict")
async def predict_disease(
    file: UploadFile = File(...)
):

    
    # FILE TYPE CHECK
  

    allowed_types = [
        "image/jpeg",
        "image/png",
        "image/jpg",
        "image/webp"
    ]

    if file.content_type not in allowed_types:

        return {

            "success": False,

            "message": "Please upload a valid leaf image."
        }


    # READ IMAGE
  

    image_data = await file.read()



    # FILE SIZE CHECK
 

    if len(image_data) > 10 * 1024 * 1024:

        return {

            "success": False,

            "message": "Image size must be less than 10MB."
        }


   
    # OPEN IMAGE
  

    try:

        image = Image.open(
            io.BytesIO(image_data)
        ).convert("RGB")

    except Exception:

        return {

            "success": False,

            "message": "Unable to read the uploaded image."
        }


  
    # PREPROCESS IMAGE
  

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)


   
    # MODEL PREDICTION
  

    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted_index = torch.max(
            probabilities,
            1
        )



    # GET PREDICTION


    predicted_class = class_names[
        predicted_index.item()
    ]


    confidence_percentage = (
        confidence.item() * 100
    )


  
    # LOW CONFIDENCE


    if confidence_percentage < 70:

        return {

            "success": True,

            "disease": "Uncertain Result",

            "confidence": round(
                confidence_percentage,
                2
            ),

            "status": "Image Needs Review",

            "riskLevel": "Unknown",

            "treatment": [

                "Do not apply treatment based on this result",

                "Capture another clear leaf image",

                "Consult an agricultural expert if symptoms continue"
            ],

            "prevention": [

                "Use good lighting",

                "Keep the leaf fully visible",

                "Avoid blurry or shadowed images"
            ],

            "imageName": file.filename,

            "imageSize": len(image_data),

            "model": "ResNet18",

            "predictionType": "ML Model",

            "recommendation": "Please upload a clearer leaf image."
        }


  
    # DISEASE INFORMATION
  

    info = DISEASE_INFO.get(

        predicted_class,

        {

            "status": "Possible Disease",

            "riskLevel": "Moderate",

            "treatment": [

                "Consult an agricultural expert",

                "Monitor the crop closely"
            ],

            "prevention": [

                "Maintain good field hygiene",

                "Monitor affected leaves"
            ]
        }
    )


    
    # MEDIUM CONFIDENCE
  

    if confidence_percentage < 85:

        info = {

            **info,

            "status": "Possible Disease",

            "riskLevel": "Moderate"
        }


    # =========================
    # FINAL RESPONSE
    # =========================

    return {

        "success": True,

        "disease": (
            predicted_class
            .replace("_", " ")
        ),

        "confidence": round(
            confidence_percentage,
            2
        ),

        "status": info["status"],

        "riskLevel": info["riskLevel"],

        "treatment": info["treatment"],

        "recommendation": info["treatment"],

        "prevention": info["prevention"],

        "imageName": file.filename,

        "imageSize": len(image_data),

        "model": "ResNet18",

        "predictionType": "ML Model"
    }