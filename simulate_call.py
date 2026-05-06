import socket
import struct
import uuid

def simulate_call(did_part="83740150", caller_num="972501234567"):
    # Create the UUID format: [DID]-2222-3333-4444-[CALLER]
    # In main.py: did_part = parts[0].lstrip("0"), caller_number = parts[-1].lstrip("0")
    
    # We need to craft a valid UUID string that fits the expected format
    # The split("-") will be used.
    # did_part (8 chars) - 2222 (4) - 3333 (4) - 4444 (4) - caller (12 chars)
    # Total hex chars: 8+4+4+4+12 = 32 (correct for UUID)
    
    padded_caller = caller_num.zfill(12)
    uuid_str = f"{did_part}-2222-3333-4444-{padded_caller}"
    print(f"Simulating call with UUID: {uuid_str}")
    
    call_uuid = uuid.UUID(uuid_str)
    
    # Header format: Kind (1 byte) + Length (2 bytes Big Endian)
    KIND_ID = 0x01
    payload = call_uuid.bytes
    header = struct.pack(">BH", KIND_ID, len(payload))
    
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("127.0.0.1", 9090))
        s.sendall(header + payload)
        print("UUID sent successfully. Check the VoiceAgent logs.")
        s.close()
    except Exception as e:
        print(f"Error: {e}. Is the VoiceAgent running on port 9090?")

if __name__ == "__main__":
    simulate_call()
