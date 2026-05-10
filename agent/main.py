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
SILENCE_THRESHOLD = 1500 # Abaissé pour capter les mots très courts comme 'כן'
SILENCE_DURATION_FRAMES = 50 # Blanc de ~1s avant de répondre
INTERRUPTION_FRAMES = 5 # Abaissé à ~100ms de voix continue avant de couper le bot

# Clients
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
mongo_client = MongoClient(os.getenv("MONGO_URI"))
db = mongo_client["leader_db"]

SYSTEM_PROMPT = """אתה מוקדן שירות במוקד Leader Taxi. 
עליך להיות מקצועי, ענייני ומהיר מאוד. הלקוחות רוצים להזמין מונית במינימום זמן.
חשוב: עבודתך מתבצעת בשני שלבים.

שלב 1: איסוף נתונים
- שאל "מאיפה האיסוף?" (עיר ורחוב).
- שאל "ולאן היעד?" (עיר ורחוב).
השתמש בפונקציה 'enregistrer_adresses' כדי לשמור את הכתובות שזיהית. אל תשאל לאישור לפני שיש לך את שתי הכתובות המלאות.

שלב 2: אימות (חובה!)
לאחר קבלת שתי הכתובות, עליך לחזור עליהן באוזני הלקוח ולבקש אישור מפורש:
"אני מסכם: איסוף מ[כתובת מוצא] ל[כתובת יעד]. זה נכון?"
- אם הלקוח עונה בחיוב ("כן", "נכון", "בסדר", "בדיוק", "סבבה", "זהו", או כל מילה נרדפת): עליך לקרוא מיד לפונקציה 'confirmer_course' כדי לבצע את ההזמנה בפועל.
- אם הלקוח עונה בשלילה ("לא", "טעות"): עליך לקרוא לפונקציה 'annuler_ou_recommencer' ולבקש מהלקוח לתקן את הכתובת.

לעולם אל תקרא לפונקציה 'confirmer_course' לפני שעשית סיכום וקיבלת אישור מפורש!

חשוב מאוד: זיהוי קולי בעברית נוטה לטעויות. תקן אותן באופן אוטומטי לפני השימוש בכלי:
- "אזון" או "אזור" או "אשזוד" -> "אשדוד".
- "בני איש" או "ביי אהזרנה" או "ביי אה זרנה" או "אהזרנה" או "ביילדות" -> "בני עי"ש".
- "כאילו מושך" או "כאילו משה" -> "קרית משה".
- "היי משה" או "חיים משה" -> "חיים משה שפירא" (אם זה נשמע כמו התחלה של רחוב).
- "שפירה" -> "שפירא", "הביטיחות" -> "הבטיחות", "רחוב הגדוד" -> "הגדוד העברי".
- אם שמעת שם של עיר ורחוב, גם אם הם נשמעים קצת משובשים, נסה לתקן אותם לערים ורחובות אמיתיים באזור המרכז/דרום (אשדוד, בני עי"ש, גדרה, רחובות).

הנחיות נוספות:
- אם הלקוח ציין שתי כתובות (מוצא ויעד) אבל אמר את שם העיר רק פעם אחת (למשל "מחיים משה שפירא להדקל באשדוד"), הנח ששני הרחובות נמצאים באותה עיר (אשדוד). אל תשאל באיזו עיר אם אפשר להסיק זאת!
- אם הלקוח אמר רחוב ולא ציין עיר בכלל, רק אז שאל אותו "באיזו עיר?" לפני שאתה מנסה להזמין.
- אל תנחש כתובות. אם הלקוח אמר משהו לא ברור, בקש ממנו לחזור שוב.
- אם הפונקציה 'confirmer_course' מחזירה שגיאה (למשל שהכתובת לא נמצאה), אל תתחיל את השיחה מהתחלה! פשוט תגיד "מצטער, לא מצאתי את הכתובת, אפשר לדייק אותה?" ותמשיך משם.
סיים את השיחה באישור קצר."""
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

def internal_verify_address(city_name: str, street_name: str):
    """Vérifie si une rue existe dans une ville via l'API Nominatim."""
    try:
        clean_street = street_name.replace("רחוב", "").strip()
        import re
        clean_street_no_num = re.sub(r'\d+', '', clean_street).strip()
        
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "street": clean_street_no_num,
            "city": city_name,
            "country": "Israel",
            "format": "json"
        }
        headers = {"User-Agent": "LeaderTaxiAgent/1.0"}
        r = requests.get(url, params=params, headers=headers, timeout=5)
        res = r.json()
        if len(res) > 0:
            return f"L'adresse {street_name} existe bien à {city_name}."
        else:
            return f"Je n'ai pas trouvé l'adresse {street_name} à {city_name}. Demande à l'utilisateur s'il est sûr de l'adresse ou s'il y a une erreur."
    except Exception as e:
        logger.error(f"Erreur vérification adresse: {e}")
        return "Impossible de vérifier l'adresse pour le moment."

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
            "name": "verifier_adresse",
            "description": "Vérifie si une rue/adresse existe dans une ville donnée. À utiliser si l'utilisateur demande explicitement de vérifier une adresse.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city_name": {"type": "string", "description": "La ville."},
                    "street_name": {"type": "string", "description": "Le nom de la rue à vérifier."}
                },
                "required": ["city_name", "street_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "enregistrer_adresses",
            "description": "À appeler pour enregistrer l'adresse de départ et de destination avant de demander confirmation à l'utilisateur. Ne déclenche pas la commande.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin_city": {"type": "string", "description": "La ville de départ."},
                    "origin_address": {"type": "string", "description": "L'adresse de départ."},
                    "destination_city": {"type": "string", "description": "La ville de destination."},
                    "destination_address": {"type": "string", "description": "L'adresse de destination."}
                },
                "required": ["origin_city", "origin_address", "destination_city", "destination_address"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "confirmer_course",
            "description": "À appeler UNIQUEMENT lorsque l'utilisateur a explicitement confirmé les adresses (ex: 'oui', 'c'est ça', 'valide'). Cette fonction déclenche la commande finale.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin_city": {"type": "string", "description": "La ville de départ confirmée."},
                    "origin_address": {"type": "string", "description": "L'adresse précise de départ confirmée."},
                    "destination_city": {"type": "string", "description": "La ville de destination confirmée."},
                    "destination_address": {"type": "string", "description": "L'adresse précise de destination confirmée."}
                },
                "required": ["origin_city", "origin_address", "destination_city", "destination_address"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "annuler_ou_recommencer",
            "description": "À appeler si l'utilisateur infirme, veut annuler ou corriger une erreur.",
            "parameters": {
                "type": "object",
                "properties": {
                    "raison": {"type": "string", "description": "Pourquoi l'utilisateur veut annuler ou recommencer"}
                },
                "required": []
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
        # Utilisation de edge-tts avec réglages pour une voix plus dynamique et moins monotone
        VOICE = "he-IL-HilaNeural" 
        # Augmentation de la vitesse (+10%) et léger ajustement du ton pour plus de naturel
        communicate = edge_tts.Communicate(text, VOICE, rate="+10%", pitch="+2Hz")
        
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
        
        logger.info(f"Audio TTS Natif (Hila) généré : {len(raw_pcm)} bytes")
        
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

async def process_audio_and_respond(audio_buffer: bytes, chat_history: list, caller_number: str, caller_name: str):
    logger.info(f"Analyse audio de {len(audio_buffer)} bytes...")
    should_hangup = False
    
    # Nettoyage de l'historique pour éviter les hallucinations dues à un contexte trop long
    if len(chat_history) > 15:
        # On garde le prompt système (index 0) et les 10 derniers messages
        chat_history[:] = [chat_history[0]] + chat_history[-10:]
    
    # 1. STT
    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wav_file:
        wav_file.setnchannels(1), wav_file.setsampwidth(2), wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(audio_buffer)
    wav_io.name = "audio.wav"
    wav_io.seek(0)
    
    try:
        transcript = await client.audio.transcriptions.create(
            model="whisper-1", 
            file=wav_io, 
            language="he",
            prompt="אשדוד, בני עיש, קרית משה, חיים משה שפירא, ירושלים, תל אביב, רחובות, גדרה"
        )
        user_text = transcript.text
        if len(user_text.strip()) < 2: return
        logger.info(f"User ({caller_name}): {user_text}")
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
                try:
                    args = json.loads(tool_call.function.arguments)
                except Exception:
                    args = {}
                logger.info(f"Appel outil {tool_call.function.name} avec {args}")
                
                if tool_call.function.name == "check_pharmacy_stock":
                    result = internal_check_pharmacy_stock(args.get("medicine_name"), args.get("city_name", "Jérusalem"))
                    chat_history.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": "check_pharmacy_stock",
                        "content": result
                    })
                elif tool_call.function.name == "verifier_adresse":
                    result = internal_verify_address(args.get("city_name"), args.get("street_name"))
                    chat_history.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": "verifier_adresse",
                        "content": result
                    })
                elif tool_call.function.name == "enregistrer_adresses":
                    chat_history.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": "enregistrer_adresses",
                        "content": "Adresses enregistrées en mémoire. Tu dois maintenant impérativement demander à l'utilisateur de confirmer ces adresses de manière claire."
                    })
                elif tool_call.function.name == "confirmer_course":
                    should_hangup = True
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
                        "name": "confirmer_course",
                        "content": result
                    })
                elif tool_call.function.name == "annuler_ou_recommencer":
                    chat_history.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": "annuler_ou_recommencer",
                        "content": "Annulation prise en compte. Demande à l'utilisateur de préciser les bonnes adresses."
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

    # 3. On retourne le texte pour le TTS
    return bot_text, should_hangup

async def handle_audiosocket(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    addr = writer.get_extra_info('peername')
    logger.info("NOUVEL APPEL RECU")
    
    chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    audio_buffer, silence_frames, noise_frames, is_speaking = bytearray(), 0, 0, False
    current_response_task = None
    response_start_time = 0
    caller_number = "Inconnu"
    
    bot_state = {"is_thinking": False, "is_speaking": False}
    last_processed_buffer = bytearray()
    
    async def pipeline(audio_data):
        bot_state["is_thinking"] = True
        bot_state["is_speaking"] = False
        try:
            bot_text, should_hangup = await process_audio_and_respond(audio_data, chat_history, caller_number, caller_name)
            if not bot_text: return
            bot_state["is_thinking"] = False
            bot_state["is_speaking"] = True
            await send_tts(bot_text, writer)
            if should_hangup:
                logger.info("Fin du process, on raccroche l'appel.")
                try:
                    writer.write(struct.pack(">BH", KIND_HANGUP, 0))
                    await writer.drain()
                except Exception:
                    pass
                writer.close()
        except asyncio.CancelledError:
            pass
        finally:
            bot_state["is_thinking"] = False
            bot_state["is_speaking"] = False
    
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
                import time
                call_id_obj = uuid.UUID(bytes=payload)
                call_id_str = str(call_id_obj)
                
                # Extraction multi-tenant
                # UUID format: [DID]-2222-3333-4444-[CALLER]
                parts = call_id_str.split("-")
                did_part = parts[0].lstrip("0")
                caller_number = parts[-1].lstrip("0")
                
                logger.info(f"Appel reçu - DID: {did_part}, Client: {caller_number}")
                
                # Identification du client
                caller_name = get_caller_identity(caller_number)
                logger.info(f"Identité identifiée : {caller_name}")

                # Identification de l'agence (Tenant)
                agency_name = "Leader Real Estate"
                try:
                    tenant = db["tenants"].find_one({"did": {"$regex": f"{did_part}$"}})
                    if tenant:
                        agency_name = tenant.get("name", agency_name)
                        logger.info(f"Agence identifiée : {agency_name}")
                except Exception as db_e:
                    logger.error(f"Erreur lookup tenant: {db_e}")

                chat_history[0]["content"] += f"\nTu es l'assistant de l'agence : {agency_name}."
                chat_history[0]["content"] += f"\nLe nom du client est : {caller_name} (numéro: {caller_number})."
                
                # Greeting in Hebrew
                greeting = f"שלום {caller_name}, מוקד {agency_name}. מאיפה לאסוף אותך ולאן היעד?"
                chat_history.append({"role": "assistant", "content": greeting})
                response_start_time = time.time()
                bot_state["is_speaking"] = True
                
                async def play_greeting():
                    try:
                        await send_tts(greeting, writer)
                    finally:
                        bot_state["is_speaking"] = False
                        
                current_response_task = asyncio.create_task(play_greeting())
                continue
                
            if kind_val == KIND_AUDIO:
                import time
                rms = compute_rms(payload)
                if rms > SILENCE_THRESHOLD:
                    noise_frames += 1
                    # On ne considère que c'est de la parole que si on a au moins X frames
                    if noise_frames >= INTERRUPTION_FRAMES:
                        if current_response_task and not current_response_task.done():
                            # Fenêtre de protection de 500ms pour éviter l'auto-coupure (écho)
                            if time.time() - response_start_time > 0.5:
                                logger.info(f"Interruption confirmée (RMS: {rms}), arrêt de la réponse. Thinking: {bot_state['is_thinking']}")
                                current_response_task.cancel()
                                current_response_task = None
                                
                                # Si le bot était en train de réfléchir (STT/LLM), on récupère l'audio précédent
                                # pour ne pas perdre le début de la phrase de l'utilisateur !
                                if bot_state["is_thinking"]:
                                    audio_buffer = last_processed_buffer + audio_buffer
                                last_processed_buffer = bytearray()
                        is_speaking, silence_frames = True, 0
                    audio_buffer.extend(payload)
                else:
                    noise_frames = 0
                    if is_speaking:
                        audio_buffer.extend(payload)
                        silence_frames += 1
                        if silence_frames > SILENCE_DURATION_FRAMES:
                            response_start_time = time.time()
                            last_processed_buffer = bytearray(audio_buffer)
                            current_response_task = asyncio.create_task(pipeline(bytes(audio_buffer)))
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
        if current_response_task and not current_response_task.done():
            current_response_task.cancel()
        writer.close()

async def main():
    server = await asyncio.start_server(handle_audiosocket, '0.0.0.0', 9090)
    logger.info('PABX Smart Agent started on :9090')
    async with server: await server.serve_forever()

if __name__ == "__main__":
    asyncio.run(main())
