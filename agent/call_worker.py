import os
import time
import logging
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CallWorker")

# --- CONFIGURATION ---
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "leader_db"
# Dossier où Asterisk guette les fichiers .call
# Dans Docker, ce sera un volume partagé
ASTERISK_OUTGOING_DIR = "/var/spool/asterisk/outgoing" 

# On s'assure que le dossier temporaire existe pour construire le fichier avant de le déplacer
TEMP_DIR = "/tmp/asterisk_calls"
os.makedirs(TEMP_DIR, exist_ok=True)
if not os.path.exists(ASTERISK_OUTGOING_DIR):
    # Fallback pour le développement local si non monté
    logger.warning(f"Dossier {ASTERISK_OUTGOING_DIR} non trouvé. Utilisation de {TEMP_DIR} pour simulation.")
    ASTERISK_OUTGOING_DIR = TEMP_DIR

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
call_queue = db["call_queue"]

def generate_call_file(phone_number, request_id):
    """
    Génère un fichier .call pour Asterisk.
    Format :
    Channel: PJSIP/NUMERO@zadarma_endpoint
    Context: taxi-arrival-notification
    Extension: s
    Priority: 1
    """
    # Nettoyage du numéro pour Zadarma (souvent format international sans + ou 00)
    clean_phone = phone_number.replace("+", "").replace(" ", "").lstrip("0")
    # Si le numéro commence par 5, on assume Israël et on ajoute 972
    if clean_phone.startswith("5"):
        clean_phone = "972" + clean_phone
        
    filename = f"call_{request_id}_{int(time.time())}.call"
    temp_path = os.path.join(TEMP_DIR, filename)
    final_path = os.path.join(ASTERISK_OUTGOING_DIR, filename)
    
    content = f"""Channel: PJSIP/{clean_phone}@zadarma_endpoint
MaxRetries: 2
RetryTime: 60
WaitTime: 30
Context: taxi-arrival-notification
Extension: s
Priority: 1
Set: CALL_REQUEST_ID={request_id}
"""
    
    try:
        with open(temp_path, "w") as f:
            f.write(content)
        
        # Déplacer le fichier vers le dossier 'outgoing' d'Asterisk est atomique (mieux que d'écrire directement dedans)
        os.rename(temp_path, final_path)
        logger.info(f"✅ Fichier .call généré pour {phone_number} -> {final_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Erreur génération fichier .call: {e}")
        return False

def process_queue():
    logger.info("Démarrage du worker de file d'attente d'appels...")
    while True:
        try:
            # On cherche un appel en attente
            pending_call = call_queue.find_one_and_update(
                {"status": "pending"},
                {"$set": {"status": "processing", "startedAt": datetime.utcnow()}},
                sort=[("createdAt", 1)]
            )
            
            if pending_call:
                phone = pending_call.get("phoneNumber")
                req_id = pending_call.get("requestId")
                
                logger.info(f"Traitement de l'appel pour {phone} (Request: {req_id})")
                
                success = generate_call_file(phone, req_id)
                
                if success:
                    call_queue.update_one(
                        {"_id": pending_call["_id"]},
                        {"$set": {"status": "completed", "completedAt": datetime.utcnow()}}
                    )
                else:
                    call_queue.update_one(
                        {"_id": pending_call["_id"]},
                        {"$set": {"status": "failed", "error": "Failed to generate call file"}}
                    )
            
            time.sleep(2) # Pause de 2 secondes entre les vérifications
            
        except Exception as e:
            logger.error(f"Erreur boucle process_queue: {e}")
            time.sleep(5)

if __name__ == "__main__":
    process_queue()
