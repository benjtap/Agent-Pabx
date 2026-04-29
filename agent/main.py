import asyncio
import logging
import struct
import io
import wave
import math
import os
import json
import requests
from dotenv import load_dotenv
from openai import AsyncOpenAI
import pydub
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
import edge_tts

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VoiceAgent")

# --- CONFIGURATION ---
KIND_HANGUP = 0x00
KIND_ID = 0x01
KIND_AUDIO = 0x10
KIND_ERROR = 0xff

SAMPLE_RATE = 8000
SILENCE_THRESHOLD = 2000
SILENCE_DURATION_FRAMES = 100

# Clients
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
mongo_client = MongoClient(os.getenv("MONGO_URI"))
db = mongo_client["leader_db"]

SYSTEM_PROMPT = """אתה מוקדן שירות במוקד Leader Taxi. 
עליך להיות מקצועי, ענייני ומהיר מאוד. הלקוחות רוצים להזמין מונית במינימום זמן.
פעל לפי השלבים הבאים:
1. שאל "מאיפה האיסוף?" (עיר ורחוב).
2. שאל "ולאן היעד?" (עיר ורחוב).
אל תבזבז זמן על שיחות חולין.
חשוב מאוד: כשאתה מעביר את שם הרחוב לכלי 'order_taxi', הקפד לתקן שגיאות כתיב נפוצות שנובעות מזיהוי קולי. למשל, במקום "שפירה" כתוב "שפירא", במקום "הביטיחות" כתוב "הבטיחות". כתוב את שם הרחוב התקין ביותר שאתה מכיר.
ברגע שיש לך את כל הפרטים (מוצא ויעד), השתמש בכלי 'order_taxi' כדי לבצע את ההזמנה. לעולם אל תפעיל את הכלי לפני שיש לך גם את כתובת האיסוף וגם את כתובת היעד במלואן. סיים את השיחה באישור קצר."""
# --- OUTILS MÉTIER (TOOLS) ---

def internal_check_pharmacy_stock(medicine_name: str, city_name: str = "Jérusalem"):
    """Appelle l'API locale du scraper Meuhedet."""
    try:
        # 1. Rechercher le médicament
        search_url = "http://localhost:5005/api/search"
        r = requests.get(search_url, params={"term": medicine_name}, timeout=5)
        meds = r.json()
        if not meds: return f"Je n'ai pas trouvé de médicament nommé {medicine_name}."
        
        # 2. Vérifier l'inventaire pour le premier résultat
        inventory_url = "http://localhost:5005/api/inventory"
        city_id = "3000" if "jérusalem" in city_name.lower() else "70" # 70 = Ashdod
        payload = {"meds": [meds[0]], "city_id": city_id}
        ri = requests.post(inventory_url, json=payload, timeout=8)
        inventory = ri.json()
        
        pharmacies = inventory.get('inventories', [])
        in_stock = [p for p in pharmacies if p.get('status') == 3]
        
        if in_stock:
            return f"Oui, {medicine_name} est en stock dans {len(in_stock)} pharmacies à {city_name}, notamment chez {in_stock[0]['name']}."
        return f"Désolé, {medicine_name} semble être en rupture de stock à {city_name}."
    except Exception as e:
        logger.error(f"Erreur outil pharmacie: {e}")
        return "Je rencontre une difficulté technique pour vérifier les pharmacies."

def get_caller_identity(phone: str):
    """Recherche le nom du contact dans MongoDB."""
    contact = db["leads"].find_one({"phone": {"$regex": phone}})
    if contact:
        return contact.get("name", "client")
    return "client"

def internal_order_taxi(origin_city: str, origin_address: str, destination_city: str, destination_address: str, caller_number: str):
    """Appelle l'API LeaderAPI pour créer une requête de taxi et déclencher le scoring."""
    try:
        # On appelle l'API locale (sur le port 8081 car network_mode: host + leaderapi port mapping)
        api_url = "http://localhost:8081/api/taxi/request"
        payload = {
            "clientPhone": caller_number,
            "originCity": origin_city,
            "originAddress": origin_address,
            "destinationCity": destination_city,
            "destinationAddress": destination_address
        }
        logger.info(f"Envoi de la commande à {api_url} de {origin_address}, {origin_city} vers {destination_address}, {destination_city}")
        r = requests.post(api_url, json=payload, timeout=10)
        
        if r.status_code == 200:
            return f"הזמנת המונית מ{origin_address} ב{origin_city} ל{destination_address} ב{destination_city} נשלחה בהצלחה. נהג יצור איתך קשר בהקדם."
        else:
            logger.error(f"API Error: {r.status_code} - {r.text}")
            return "מצטער, חלה שגיאה בחיבור למערכת ההזמנות."
    except Exception as e:
        logger.error(f"Erreur outil taxi: {e}")
        return "מצטער, אני נתקל בקושי טכני בהזמנת המונית."

TOOLS_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "check_pharmacy_stock",
            "description": "Vérifie la disponibilité d'un médicament dans une ville",
            "parameters": {
                "type": "object",
                "properties": {
                    "medicine_name": {"type": "string", "description": "Le nom du médicament (ex: Acamol, Galvus)"},
                    "city_name": {"type": "string", "description": "La ville (ex: Jérusalem, Ashdod)"}
                },
                "required": ["medicine_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "order_taxi",
            "description": "Commande un taxi pour le client en enregistrant son point de départ et sa destination.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin_city": {"type": "string", "description": "La ville de départ (ex: Ashdod, Jérusalem)."},
                    "origin_address": {"type": "string", "description": "L'adresse précise de départ."},
                    "destination_city": {"type": "string", "description": "La ville de destination."},
                    "destination_address": {"type": "string", "description": "L'adresse précise de destination."}
                },
                "required": ["origin_city", "origin_address", "destination_city", "destination_address"]
            }
        }
    }
]

# --- PIPELINE AUDIO ---

def compute_rms(pcm_data: bytes) -> float:
    count = len(pcm_data) // 2
    if count == 0: return 0.0
    shorts = struct.unpack(f"<{count}h", pcm_data)
    sum_sq = sum(s * s for s in shorts)
    return math.sqrt(sum_sq / count)

async def send_tts(text: str, writer: asyncio.StreamWriter):
    """Génère le TTS avec un accent israélien natif (edge-tts) et l'envoie via AudioSocket."""
    try:
        # Utilisation de edge-tts pour un accent hébreu natif sans accent américain
        VOICE = "he-IL-AvriNeural" # "he-IL-HilaNeural" pour une voix féminine
        communicate = edge_tts.Communicate(text, VOICE)
        
        # On récupère l'audio en mémoire
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]

        # Conversion via pydub pour correspondre au format AudioSocket (8000Hz, Mono, S16LE)
        audio_segment = pydub.AudioSegment.from_mp3(io.BytesIO(audio_data))
        audio_segment = audio_segment.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
        
        raw_io = io.BytesIO()
        audio_segment.export(raw_io, format="s16le")
        raw_pcm = raw_io.getvalue()
        
        logger.info(f"Audio TTS (Israélien) généré : {len(raw_pcm)} bytes")
        
        chunk_size = 320
        for i in range(0, len(raw_pcm), chunk_size):
            chunk = raw_pcm[i:i+chunk_size]
            if len(chunk) < chunk_size:
                chunk += b'\x00' * (chunk_size - len(chunk))
            
            header = struct.pack(">BH", KIND_AUDIO, len(chunk))
            writer.write(header + chunk)
            await writer.drain()
            await asyncio.sleep(0.020)
        logger.info("Fin de la transmission audio")
    except Exception as e:
        logger.error(f"TTS Error: {e}")

async def process_audio_and_respond(audio_buffer: bytes, writer: asyncio.StreamWriter, chat_history: list, caller_number: str):
    logger.info(f"Analyse audio de {len(audio_buffer)} bytes...")
    
    # 1. STT
    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wav_file:
        wav_file.setnchannels(1), wav_file.setsampwidth(2), wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(audio_buffer)
    wav_io.name = "audio.wav"
    wav_io.seek(0)
    
    try:
        transcript = await client.audio.transcriptions.create(model="whisper-1", file=wav_io, language="he")
        user_text = transcript.text
        if len(user_text.strip()) < 2: return
        logger.info(f"User: {user_text}")
        chat_history.append({"role": "user", "content": user_text})
    except Exception as e:
        logger.error(f"STT Error: {e}"); return

    # 2. LLM avec Tools
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=chat_history,
            tools=TOOLS_DEFINITION,
            tool_choice="auto",
            parallel_tool_calls=False
        )
        
        message = response.choices[0].message
        
        # Gestion des appels d'outils
        if message.tool_calls:
            chat_history.append(message)
            for tool_call in message.tool_calls:
                args = json.loads(tool_call.function.arguments)
                logger.info(f"Appel outil {tool_call.function.name} avec {args}")
                
                if tool_call.function.name == "check_pharmacy_stock":
                    result = internal_check_pharmacy_stock(args.get("medicine_name"), args.get("city_name", "Jérusalem"))
                    chat_history.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": "check_pharmacy_stock",
                        "content": result
                    })
                elif tool_call.function.name == "order_taxi":
                    result = internal_order_taxi(
                        args.get("origin_city"),
                        args.get("origin_address"),
                        args.get("destination_city"),
                        args.get("destination_address"),
                        caller_number
                    )
                    chat_history.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": "order_taxi",
                        "content": result
                    })
            
            # Deuxième passage pour générer la réponse finale
            response = await client.chat.completions.create(model="gpt-4o-mini", messages=chat_history)
            bot_text = response.choices[0].message.content
        else:
            bot_text = message.content

        logger.info(f"Agent répond: {bot_text}")
        chat_history.append({"role": "assistant", "content": bot_text})
    except Exception as e:
        logger.error(f"LLM Error: {e}"); bot_text = "מצטער, חלה שגיאה."

    # 3. TTS
    await send_tts(bot_text, writer)

async def handle_audiosocket(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    addr = writer.get_extra_info('peername')
    logger.info("NOUVEL APPEL RECU")
    
    chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    audio_buffer, silence_frames, is_speaking = bytearray(), 0, False
    current_response_task = None
    caller_number = "Inconnu"
    
    try:
        while True:
            # Format standard AudioSocket : Kind (1 octet) + Length (2 octets Big Endian)
            header = await reader.readexactly(3)
            kind_val, payload_len = struct.unpack(">BH", header)
            
            payload = await reader.readexactly(payload_len)
            
            if kind_val == KIND_HANGUP:
                logger.info("Appel terminé (Hangup)")
                break
                
            if kind_val == KIND_ID:
                import uuid
                call_id_obj = uuid.UUID(bytes=payload)
                call_id_str = str(call_id_obj)
                # L'UUID a été formaté avec le numéro à la fin
                num = call_id_str.split("-")[-1].lstrip("0")
                if num:
                    caller_number = num
                logger.info(f"Appel reçu, UUID: {call_id_str}, Numéro: {caller_number}")
                
                chat_history[0]["content"] += f"\nLe numéro de téléphone du client appelant est : {caller_number}."
                
                # Greeting in Hebrew
                greeting = "מוקד לידר טקסי שלום, מאיפה לאסוף אותך ולאן היעד?"
                chat_history.append({"role": "assistant", "content": greeting})
                asyncio.create_task(send_tts(greeting, writer))
                continue
                
            if kind_val == KIND_AUDIO:
                rms = compute_rms(payload)
                if rms > SILENCE_THRESHOLD:
                    if current_response_task and not current_response_task.done():
                        logger.info(f"Bruit ignoré pendant la réponse (RMS: {rms})")
                        # current_response_task.cancel() # Désactivé temporairement pour éviter les coupures
                    is_speaking, silence_frames = True, 0
                    audio_buffer.extend(payload)
                else:
                    if is_speaking:
                        audio_buffer.extend(payload)
                        silence_frames += 1
                        if silence_frames > SILENCE_DURATION_FRAMES:
                            current_response_task = asyncio.create_task(process_audio_and_respond(bytes(audio_buffer), writer, chat_history, caller_number))
                            audio_buffer, is_speaking, silence_frames = bytearray(), False, 0
            elif kind_val == KIND_ERROR:
                logger.error("Erreur reçue d'AudioSocket")
                break
            else:
                logger.warning(f"Kind inconnu reçu: {kind_val} (Longueur: {payload_len})")
    except asyncio.IncompleteReadError:
        logger.info("Connexion fermée par le client.")
    except (ConnectionResetError, BrokenPipeError):
        logger.error("ERREUR Pipeline: Connection lost")
    finally:
        writer.close()

async def main():
    server = await asyncio.start_server(handle_audiosocket, '0.0.0.0', 9090)
    logger.info('PABX Smart Agent started on :9090')
    async with server: await server.serve_forever()

if __name__ == "__main__":
    asyncio.run(main())
