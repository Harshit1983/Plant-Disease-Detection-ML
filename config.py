import os

# Disable oneDNN custom ops to prevent MKL memory allocation errors on Windows CPU
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VALID_DIR = os.path.join(DATASET_DIR, "valid")
TEST_DIR = os.path.join(DATASET_DIR, "test")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# Raw Dataset Path (PlantVillage repository folder)
RAW_DATASET_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "PlantVillage"))

# Artifact Paths
MODEL_PATH = os.path.join(MODELS_DIR, "plant_disease_model.keras")
CLASS_INDICES_PATH = os.path.join(MODELS_DIR, "class_indices.json")
TRAINING_HISTORY_PATH = os.path.join(REPORTS_DIR, "training_history.json")
SPLIT_REPORT_PATH = os.path.join(REPORTS_DIR, "dataset_split_report.json")

# Model Training & Inference Parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 16))
EPOCHS = int(os.environ.get("EPOCHS", 10))
FINE_TUNE_EPOCHS = int(os.environ.get("FINE_TUNE_EPOCHS", 5))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 1e-3))
FINE_TUNE_LEARNING_RATE = float(os.environ.get("FINE_TUNE_LEARNING_RATE", 1e-4))
RANDOM_SEED = 42

# Service settings
SERVICE_HOST = os.environ.get("HOST", "0.0.0.0")
SERVICE_PORT = int(os.environ.get("PORT", 8000))

# 15 Standard Plant Pathology Target Classes
CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

NUM_CLASSES = len(CLASS_NAMES)

# Detailed Agronomic Diagnostic & Pathology Knowledge Base
DISEASE_PATHOLOGY_INFO = {
    "Pepper__bell___Bacterial_spot": {
        "crop": "Pepper",
        "disease_name": "Bacterial Spot",
        "scientific_name": "Xanthomonas euvesicatoria",
        "severity": "Moderate to High",
        "symptoms": "Small, water-soaked brown lesions on lower leaf surfaces that turn necrotic with yellow halos.",
        "environmental_factors": "High humidity (>85%), warm temperatures (24-30°C), overhead rainfall or splashing irrigation.",
        "organic_treatment": "Apply copper octanoate or copper hydroxide fungicides. Prune affected bottom foliage to enhance air movement.",
        "chemical_treatment": "Foliar sprays containing copper sulfate coupled with mancozeb applied early in infection cycles.",
        "prevention": "Use certified disease-free seeds, avoid overhead watering, rotate crops with non-solanaceous plants every 2-3 years."
    },
    "Pepper__bell___healthy": {
        "crop": "Pepper",
        "disease_name": "Healthy Foliage",
        "scientific_name": "Capsicum annuum",
        "severity": "None",
        "symptoms": "Vibrant green, firm foliage with uniform surface color and clean venation.",
        "environmental_factors": "Optimal growing conditions: 20-28°C, well-draining soil, adequate sunlight.",
        "organic_treatment": "No disease intervention needed. Maintain balanced compost tea or organic nitrogen feeding.",
        "chemical_treatment": "None required.",
        "prevention": "Maintain steady drip irrigation, mulch around base to preserve root moisture, monitor regularly for aphids."
    },
    "Potato___Early_blight": {
        "crop": "Potato",
        "disease_name": "Early Blight",
        "scientific_name": "Alternaria solani",
        "severity": "Moderate",
        "symptoms": "Distinct concentric rings forming 'target' spots on older leaves, surrounded by yellow chlorotic tissue.",
        "environmental_factors": "Alternating wet and dry weather, temperature range 24-29°C, nitrogen deficiency stress.",
        "organic_treatment": "Apply neem oil or bio-fungicides containing Bacillus subtilis. Destroy infected lower leaves promptly.",
        "chemical_treatment": "Chlorothalonil, azoxystrobin, or difenoconazole spray applied upon first symptom sighting.",
        "prevention": "Ensure adequate potassium/nitrogen nutrition, practice 3-year crop rotation, space plants for canopy ventilation."
    },
    "Potato___Late_blight": {
        "crop": "Potato",
        "disease_name": "Late Blight",
        "scientific_name": "Phytophthora infestans",
        "severity": "Critical / Severe",
        "symptoms": "Rapidly expanding dark brown to black water-soaked lesions with whitish mold spores on leaf undersides in wet conditions.",
        "environmental_factors": "Cool, high humidity (>90%) with rain/fog, temperatures 15-22°C. Capable of destroying entire crop within days.",
        "organic_treatment": "Immediate copper-based bio-fungicide spray. Destroy infected vines completely (do not compost).",
        "chemical_treatment": "Systemic fungicides containing metalaxyl, cymoxanil, or fluazinam.",
        "prevention": "Plant certified blight-resistant seed tubers, hill up soil over tubers, eliminate volunteer potato plants."
    },
    "Potato___healthy": {
        "crop": "Potato",
        "disease_name": "Healthy Foliage",
        "scientific_name": "Solanum tuberosum",
        "severity": "None",
        "symptoms": "Clean, lush dark green compound leaves with no necrotic spots or margin curling.",
        "environmental_factors": "Ideal soil temp 15-20°C, moist well-aerated sandy loam.",
        "organic_treatment": "No treatment required. Maintain routine weed management and hilling.",
        "chemical_treatment": "None required.",
        "prevention": "Avoid excess waterlogging, ensure good hill height, test soil nutrients annually."
    },
    "Tomato_Bacterial_spot": {
        "crop": "Tomato",
        "disease_name": "Bacterial Spot",
        "scientific_name": "Xanthomonas vesicatoria",
        "severity": "High",
        "symptoms": "Dark brown, greasy-looking irregular spots on leaves that may coalesce causing leaf drop and defoliation.",
        "environmental_factors": "Warm temperatures (24-30°C) with rain, dew, or splash sprinkler irrigation.",
        "organic_treatment": "Copper soap spray applied every 7-10 days. Remove severely infected plants.",
        "chemical_treatment": "Copper hydroxide combined with mancozeb or bactericides containing streptomycin.",
        "prevention": "Sanitize equipment, avoid working in field when plants are wet, use drip line irrigation."
    },
    "Tomato_Early_blight": {
        "crop": "Tomato",
        "disease_name": "Early Blight",
        "scientific_name": "Alternaria solani",
        "severity": "Moderate to High",
        "symptoms": "Concentric rings ('bullseye' pattern) starting on bottom mature leaves, progressing upward with yellowing leaves.",
        "environmental_factors": "High humidity, warm humid days followed by heavy dew.",
        "organic_treatment": "Copper octanoate or copper-based bio-fungicides. Stake and prune low-hanging branches.",
        "chemical_treatment": "Protectant fungicides like chlorothalonil or azoxystrobin applied every 7 to 14 days.",
        "prevention": "Mulch beds to stop soil splash, practice 3-year rotation away from nightshades, ensure proper spacing."
    },
    "Tomato_Late_blight": {
        "crop": "Tomato",
        "disease_name": "Late Blight",
        "scientific_name": "Phytophthora infestans",
        "severity": "Critical",
        "symptoms": "Large, irregular greasy dark patches on stems and leaves, rapidly turning brown and dry. White fungal fuzz on undersides.",
        "environmental_factors": "Cool, humid weather (15-22°C) with fog or frequent rain.",
        "organic_treatment": "Copper fungicides applied proactively. Remove infected foliage immediately and bag for disposal.",
        "chemical_treatment": "Systemic fungicides with dimethomorph, cyazofamid, or mandipropamid.",
        "prevention": "Destroy wild nightshade weeds, choose resistant cultivars (e.g. Iron Lady, Defiant PHR)."
    },
    "Tomato_Leaf_Mold": {
        "crop": "Tomato",
        "disease_name": "Leaf Mold",
        "scientific_name": "Passalora fulva",
        "severity": "Moderate",
        "symptoms": "Pale greenish-yellow spots on upper leaf surfaces; olive-green to velvet brown mold growth on lower leaf surfaces.",
        "environmental_factors": "High relative humidity (>85%) in greenhouses or poorly ventilated high tunnels.",
        "organic_treatment": "Sulfur or copper-based fungicides. Increase greenhouse fan ventilation and lower night humidity.",
        "chemical_treatment": "Fungicides containing chlorothalonil, difenoconazole, or potassium bicarbonate.",
        "prevention": "Maximize air circulation through pruning, space plants, keep greenhouse humidity under 85%."
    },
    "Tomato_Septoria_leaf_spot": {
        "crop": "Tomato",
        "disease_name": "Septoria Leaf Spot",
        "scientific_name": "Septoria lycopersici",
        "severity": "Moderate to High",
        "symptoms": "Numerous small circular spots (1-3mm) with dark brown margins and tan/grey centers filled with tiny black specks.",
        "environmental_factors": "Warm, wet conditions (20-25°C) with extended leaf wetness.",
        "organic_treatment": "Apply copper octanoate or liquid copper fungicide. Remove infected lower leaves.",
        "chemical_treatment": "Foliar application of chlorothalonil, mancozeb, or copper compounds.",
        "prevention": "Mulch soil surface, rotate crops, prune suckers to improve airflow, avoid overhead sprinklers."
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "crop": "Tomato",
        "disease_name": "Two-Spotted Spider Mite Damage",
        "scientific_name": "Tetranychus urticae",
        "severity": "Moderate",
        "symptoms": "Fine yellow stippling or bronzing on upper leaf surface, fine silk webbing visible under leaves.",
        "environmental_factors": "Hot, dry conditions (>30°C, low humidity), dusty field borders.",
        "organic_treatment": "Spray insecticidal soap, neem oil, or introduce predatory mites (Phytoseiulus persimilis).",
        "chemical_treatment": "Miticides containing abamectin, bifenazate, or spiromesifen.",
        "prevention": "Maintain soil moisture, reduce dust along farm paths, spray leaves with water jets to dislodge colonies."
    },
    "Tomato__Target_Spot": {
        "crop": "Tomato",
        "disease_name": "Target Spot",
        "scientific_name": "Corynespora cassiicola",
        "severity": "Moderate",
        "symptoms": "Small brown pinpoint lesions expanding into target-like brown spots with light brown centers.",
        "environmental_factors": "High humidity, temp range 20-30°C.",
        "organic_treatment": "Bio-fungicides based on Bacillus amyloliquefaciens or copper sprays.",
        "chemical_treatment": "Fungicides with chlorothalonil, pyraclostrobin, or fluxapyroxad.",
        "prevention": "Avoid dense canopies, destroy crop residues post-harvest, rotate crops."
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "crop": "Tomato",
        "disease_name": "Yellow Leaf Curl Virus",
        "scientific_name": "Begomovirus (TYLCV)",
        "severity": "High / Severe",
        "symptoms": "Severe stunting, erect cupped leaves curling upward with prominent yellowing between veins.",
        "environmental_factors": "Vectored by Bemisia tabaci whiteflies in warm sunny weather.",
        "organic_treatment": "No cure once infected. Control whitefly vectors using yellow sticky traps, insecticidal soaps, and horticultural oils.",
        "chemical_treatment": "Imidacloprid, thiamethoxam, or acetamiprid to manage whitefly populations.",
        "prevention": "Use fine insect netting, remove infected plants immediately, plant resistant hybrids."
    },
    "Tomato__Tomato_mosaic_virus": {
        "crop": "Tomato",
        "disease_name": "Tomato Mosaic Virus",
        "scientific_name": "Tobamovirus (ToMV)",
        "severity": "High",
        "symptoms": "Light and dark green mosaic mottle patterns on leaves, leaf distortion (fern-like appearance), stunted growth.",
        "environmental_factors": "Easily spread via mechanical contact, tools, hands, seed contamination.",
        "organic_treatment": "Remove and destroy infected plants immediately. Wash hands and tools with 20% milk or trisodium phosphate.",
        "chemical_treatment": "No chemical virucide available.",
        "prevention": "Disinfect tools between plants, avoid tobacco use near plants, use virus-free seeds."
    },
    "Tomato_healthy": {
        "crop": "Tomato",
        "disease_name": "Healthy Foliage",
        "scientific_name": "Solanum lycopersicum",
        "severity": "None",
        "symptoms": "Deep green compound foliage with clean serrated margins and robust stem structure.",
        "environmental_factors": "Optimal sunlight 6-8 hrs, consistent soil moisture.",
        "organic_treatment": "No disease intervention needed. Maintain organic compost tea fertilizing.",
        "chemical_treatment": "None required.",
        "prevention": "Stake plants for support, maintain mulch layer, prune lower leaves to prevent splash."
    }
}
