import os
import time
import asyncio
import websockets
import json
import base64
from flask import Flask, request, send_from_directory
from flask_sock import Sock
from twilio.twiml.voice_response import VoiceResponse, Start
from twilio.rest import Client
import logging
from threading import Thread
import audioop
import numpy as np
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
sock = Sock(app)

STATIC_AUDIO_DIR = os.path.join(app.root_path, 'static', 'audio')
os.makedirs(STATIC_AUDIO_DIR, exist_ok=True)

# Twilio configuration
TWILIO_ACCOUNT_SID = "xxxxxxxx"
TWILIO_AUTH_TOKEN = "xxxxxxx"
TWILIO_PHONE_NUMBER = "xxxxxxxxx"
TO_NUMBER = "xxxxxxxxx"

# WebSocket configuration
WS_SERVER_URL = "ws://localhost:8765"

# Active calls storage
active_calls = {}

class AudioProcessor:
    def __init__(self):
        self.twilio_rate = 8000
        self.target_rate = 16000
        self.prev_state = None
    
    def process_mulaw(self, audio_data):
        try:
            # Convert mulaw to PCM
            pcm_audio = audioop.ulaw2lin(audio_data, 2)
            
            # Resample from 8kHz to 16kHz
            resampled, self.prev_state = audioop.ratecv(
                pcm_audio, 
                2,  # sample width
                1,  # channels 
                self.twilio_rate,
                self.target_rate,
                self.prev_state
            )
            
            return resampled
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return None

class WebSocketBridge:
    def __init__(self, call_sid):
        self.call_sid = call_sid
        self.server_ws = None
        self.is_connected = False
        self.is_playing = False
        self.audio_processor = AudioProcessor()
        self.twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        self.static_folder = os.path.join(os.getcwd(), 'static', 'audio')
        os.makedirs(self.static_folder, exist_ok=True)
        self.loop = asyncio.new_event_loop()
        self.current_call = None
        self.stream_active = True
    async def connect_to_server(self):
        """Connect to AI server websocket"""
        try:
            self.server_ws = await websockets.connect(
                WS_SERVER_URL,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=None
            )
            self.is_connected = True
            logger.info(f"Connected to AI server for call {self.call_sid}")
            
            # Store current call for status checking
            self.current_call = await asyncio.to_thread(
                self.twilio_client.calls(self.call_sid).fetch
            )
            
            await self.handle_server_messages()
        except Exception as e:
            logger.error(f"Error connecting to server: {e}")

    async def handle_server_messages(self):
        """Handle messages from AI server"""
        try:
            async for message in self.server_ws:
                data = json.loads(message)
                
                if data['type'] == 'tts_file':
                    # Check call is still active before playing
                    if await self.is_call_active():
                        await self.play_tts_audio(data['filepath'], data['duration'], data['text'])
                elif data['type'] == 'transcription':
                    logger.info(f"Transcription: {data['text']}")
        except Exception as e:
            logger.error(f"Error handling server messages: {e}")
            self.is_connected = False
    async def is_call_active(self):
        """Check if call is still active"""
        try:
            self.current_call = await asyncio.to_thread(
                self.twilio_client.calls(self.call_sid).fetch
            )
            return self.current_call.status in ['in-progress', 'ringing']
        except Exception:
            return False

    async def play_tts_audio(self, filepath, duration, text):
        """Stream TTS audio to Twilio call"""
        try:
            if not await self.is_call_active():
                logger.info("Call no longer active, skipping TTS")
                return
                
            logger.info(f"Playing TTS: {text}")
            
            filename = f"tts_{time.time_ns()}.wav"
            local_path = os.path.join(self.static_folder, filename)
            
            # Copy file to static folder
            import shutil
            shutil.copy2(filepath, local_path)
            
            # Create continuous stream TwiML
            response = VoiceResponse()
            response.play(f"https://7ee8-14-195-129-246.ngrok-free.app/static/audio/{filename}")
            
            # Keep stream alive
            start = Start()
            start.stream(
                name='inbound_track',
                url=f'wss://7ee8-14-195-129-246.ngrok-free.app/stream',
                track="inbound",
                format="mulaw"
            )
            response.append(start)
            
            # Update call with new TwiML
            self.is_playing = True
            await self.update_call(str(response))
            
            # Wait for audio duration
            await asyncio.sleep(duration)
            
        except Exception as e:
            logger.error(f"Error playing TTS: {e}")
        finally:
            self.is_playing = False
            # Cleanup files
            try:
                if os.path.exists(filepath):
                    os.unlink(filepath)
                if os.path.exists(local_path):
                    os.unlink(local_path)
            except Exception as e:
                logger.error(f"Error cleaning up files: {e}")

    async def update_call(self, twiml):
        """Update ongoing call with new TwiML"""
        try:
            if await self.is_call_active():
                await asyncio.to_thread(
                    self.twilio_client.calls(self.call_sid).update,
                    twiml=twiml
                )
        except Exception as e:
            logger.error(f"Error updating call: {e}")

    async def process_audio(self, audio_data):
        """Process and send audio to server"""
        try:
            if not self.is_connected:
                return
                
            # Process audio
            processed_audio = self.audio_processor.process_mulaw(audio_data)
            
            if processed_audio:
                # Send to server
                await self.server_ws.send(json.dumps({
                    'type': 'audio',
                    'data': base64.b64encode(processed_audio).decode('utf-8')
                }))
                
        except Exception as e:
            logger.error(f"Error processing audio: {e}")

    async def cleanup(self):
        """Cleanup resources"""
        logger.info(f"Cleaning up call {self.call_sid}")
        self.is_connected = False
        if self.server_ws:
            await self.server_ws.close()

@app.route("/voice", methods=['POST'])
def voice():
    """Handle incoming calls"""
    try:
        response = VoiceResponse()
        call_sid = request.values.get('CallSid')
        logger.info(f"New call initiated: {call_sid}")
        
        # Setup persistent media stream
        start = Start()
        start.stream(
            name='inbound_track',
            url=f'wss://7ee8-14-195-129-246.ngrok-free.app/stream',
            track="inbound",
            format="mulaw"
        )
        response.append(start)
        
        # Add a long pause to keep the call active
        response.pause(length=3600)  # Keep connection for up to an hour
        
        bridge = WebSocketBridge(call_sid)
        active_calls[call_sid] = bridge
        
        # Start server connection in separate thread
        def start_websocket():
            asyncio.set_event_loop(bridge.loop)
            bridge.loop.run_until_complete(bridge.connect_to_server())
            bridge.loop.run_forever()
        
        Thread(target=start_websocket, daemon=True).start()
        
        return str(response)
    except Exception as e:
        logger.error(f"Error in voice webhook: {e}")
        return str(VoiceResponse())
@sock.route('/stream')
def stream(ws):
    """Handle Twilio media stream"""
    bridge = None
    try:
        # Handle connection messages
        connect_msg = json.loads(ws.receive())
        start_msg = json.loads(ws.receive())
        
        if start_msg.get('event') == 'start':
            call_sid = start_msg.get('start', {}).get('callSid')
            if not call_sid:
                return
                
            bridge = active_calls.get(call_sid)
            if bridge:
                logger.info(f"Media stream started for call {call_sid}")
                
                # Process stream continuously
                while bridge.stream_active:
                    try:
                        message = ws.receive()
                        if not message:
                            continue
                            
                        data = json.loads(message)
                        if data.get('event') == 'media':
                            media = data.get('media', {})
                            payload = media.get('payload')
                            track = media.get('track')
                            
                            if payload and track == 'inbound':
                                audio_data = base64.b64decode(payload)
                                asyncio.run_coroutine_threadsafe(
                                    bridge.process_audio(audio_data),
                                    bridge.loop
                                )
                        
                        elif data.get('event') == 'stop':
                            bridge.stream_active = False
                            break
                            
                    except Exception as e:
                        logger.error(f"Error in stream processing: {e}")
                        if "WebSocket connection is closed" in str(e):
                            break
                
    except Exception as e:
        logger.error(f"Error in stream handler: {e}")
    finally:
        if bridge:
            bridge.stream_active = False
@app.route("/status", methods=['POST'])
def status():
    """Handle call status changes"""
    try:
        call_sid = request.values.get('CallSid')
        call_status = request.values.get('CallStatus')
        
        if call_status in ['completed', 'failed', 'busy', 'no-answer', 'canceled']:
            if call_sid in active_calls:
                bridge = active_calls[call_sid]
                bridge.stream_active = False
                asyncio.run_coroutine_threadsafe(
                    bridge.server_ws.close() if bridge.server_ws else None,
                    bridge.loop
                )
                del active_calls[call_sid]
        
        return '', 200
    except Exception as e:
        logger.error(f"Error in status webhook: {e}")
        return '', 500

def start_client():
    """Initiate outbound call"""
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        logger.info(f"Making call to {TO_NUMBER}")
        
        call = client.calls.create(
            to=TO_NUMBER,
            from_=TWILIO_PHONE_NUMBER,
            url=f"https://7ee8-14-195-129-246.ngrok-free.app/voice",    
            record=False
        )
        
        logger.info(f"Call initiated with SID: {call.sid}")
        return call.sid
    except Exception as e:
        logger.error(f"Error making call: {e}")
        return None

if __name__ == "__main__":
    logger.info("Starting Twilio bridge server...")
    
    def make_initial_call():
        time.sleep(2)
        start_client()
    
    Thread(target=make_initial_call).start()
    app.run(host='0.0.0.0', port=5000)
    
    
    
    
