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

SYSTEM_PROMPT = """Tu es l'assistant vocal intelligent de Leader. 
Tes réponses doivent être TRÈS courtes, naturelles et chaleureuses. 
Tu peux vérifier les stocks de médicaments dans les pharmacies Meuhedet.
Si tu as besoin de faire une recherche, utilise les outils à ta disposition.
Ne lis pas d'adresses complètes ou de longs IDs au téléphone, donne juste l'essentiel."""

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
    }
]

# --- PIPELINE AUDIO ---

def compute_rms(pcm_data: bytes) -> float:
    count = len(pcm_data) // 2
    if count == 0: return 0.0
    shorts = struct.unpack(f"<{count}h", pcm_data)
    sum_sq = sum(s * s for s in shorts)
    return math.sqrt(sum_sq / count)

async def process_audio_and_respond(audio_buffer: bytes, writer: asyncio.StreamWriter, chat_history: list):
    logger.info(f"Analyse audio de {len(audio_buffer)} bytes...")
    
    # 1. STT
    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wav_file:
        wav_file.setnchannels(1), wav_file.setsampwidth(2), wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(audio_buffer)
    wav_io.name = "audio.wav"
    wav_io.seek(0)
    
    try:
        transcript = await client.audio.transcriptions.create(model="whisper-1", file=wav_io, language="fr")
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
            tool_choice="auto"
        )
        
        message = response.choices[0].message
        
        # Gestion des appels d'outils
        if message.tool_calls:
            for tool_call in message.tool_calls:
                args = json.loads(tool_call.function.arguments)
                logger.info(f"Appel outil {tool_call.function.name} avec {args}")
                
                if tool_call.function.name == "check_pharmacy_stock":
                    result = internal_check_pharmacy_stock(args.get("medicine_name"), args.get("city_name", "Jérusalem"))
                    chat_history.append(message)
                    chat_history.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": "check_pharmacy_stock",
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
        logger.error(f"LLM Error: {e}"); bot_text = "Désolé, j'ai une petite erreur."

    # 3. TTS
    try:
        audio_stream = await client.audio.speech.create(model="tts-1", voice="alloy", input=bot_text, response_format="mp3")
        audio_segment = pydub.AudioSegment.from_mp3(io.BytesIO(audio_stream.content))
        audio_segment = audio_segment.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
        
        # Exporter explicitement en s16le (Signed 16-bit Little Endian) = SLIN Asterisk
        raw_io = io.BytesIO()
        audio_segment.export(raw_io, format="s16le")
        raw_pcm = raw_io.getvalue()
        
        logger.info(f"Audio TTS généré : {len(raw_pcm)} bytes (Format SLIN Asterisk)")
        
        chunk_size = 320
        for i in range(0, len(raw_pcm), chunk_size):
            chunk = raw_pcm[i:i+chunk_size]
            # Assurer que le chunk fait exactement la taille attendue si on est à la fin (padding avec des zéros)
            if len(chunk) < chunk_size:
                chunk += b'\x00' * (chunk_size - len(chunk))
            
            # Format: KIND (1 byte) + Length (2 bytes Big Endian) + Audio Payload (320 bytes)
            header = struct.pack(">BH", KIND_AUDIO, len(chunk))
            writer.write(header + chunk)
            await writer.drain()
            # Délai 20ms pour Asterisk
            await asyncio.sleep(0.020)
        logger.info("Fin de la transmission audio")
    except (ConnectionResetError, BrokenPipeError):
        logger.error("ERREUR Pipeline: Connection lost")
    except asyncio.CancelledError:
        logger.info("TTS interrompu.")
    except Exception as e:
        logger.error(f"TTS Error: {e}")

async def handle_audiosocket(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    addr = writer.get_extra_info('peername')
    logger.info("NOUVEL APPEL RECU")
    
    chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    audio_buffer, silence_frames, is_speaking = bytearray(), 0, False
    current_response_task = None
    
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
                # UUID reçu, on peut le loguer si besoin
                continue
                
            if kind_val == KIND_AUDIO:
                rms = compute_rms(payload)
                if rms > SILENCE_THRESHOLD:
                    if current_response_task and not current_response_task.done():
                        current_response_task.cancel()
                    is_speaking, silence_frames = True, 0
                    audio_buffer.extend(payload)
                else:
                    if is_speaking:
                        audio_buffer.extend(payload)
                        silence_frames += 1
                        if silence_frames > SILENCE_DURATION_FRAMES:
                            current_response_task = asyncio.create_task(process_audio_and_respond(bytes(audio_buffer), writer, chat_history))
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
