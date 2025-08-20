import os
import time
import asyncio
import websockets
import json
import base64
from flask import Flask, request, send_from_directory, jsonify
from flask_sock import Sock
from twilio.twiml.voice_response import VoiceResponse, Start, Say, Pause, Connect, Stream
from twilio.rest import Client
import logging
from threading import Thread
import audioop
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
sock = Sock(app)

# Configuration - UPDATE THESE
TWILIO_ACCOUNT_SID = "ACb97e0b13566b8e198ee2c67d32247bcb"
TWILIO_AUTH_TOKEN = "b6eb05cd4477536c9e8c525554406b87"
TWILIO_PHONE_NUMBER = "+15134961966"
TO_NUMBER = "+918220538680"

# WebSocket configuration
WS_SERVER_URL = "ws://localhost:8765"
NGROK_URL = "https://7c3cb3a9fe7d.ngrok-free.app"

active_calls = {}
call_states = {}


class AudioProcessor:
    def __init__(self):
        self.twilio_rate = 8000
        self.target_rate = 16000
        self.prev_state = None

    def process_mulaw(self, audio_data):
        try:
            pcm_audio = audioop.ulaw2lin(audio_data, 2)
            resampled, self.prev_state = audioop.ratecv(
                pcm_audio, 2, 1, self.twilio_rate, self.target_rate, self.prev_state
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
        self.audio_processor = AudioProcessor()
        self.twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        self.loop = asyncio.new_event_loop()
        self.stream_active = True
        self.first_message_sent = False
        self.connection_established = False

    async def connect_to_server(self):
        max_retries = 5
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1} to connect to AI server")
                self.server_ws = await websockets.connect(
                    WS_SERVER_URL,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5
                )
                self.is_connected = True
                self.connection_established = True
                logger.info(f"✅ Connected to AI server for call {self.call_sid}")

                # Send initial greeting trigger
                await asyncio.sleep(2)  # Wait for stream to be ready
                await self.trigger_initial_greeting()

                await self.handle_server_messages()
                break

            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                else:
                    logger.error("Failed to connect to AI server after all attempts")

    async def trigger_initial_greeting(self):
        """Trigger the initial AI greeting"""
        try:
            if self.server_ws and self.is_connected:
                # Send a dummy audio packet to trigger the conversation
                dummy_audio = b'\x00' * 640  # 40ms of silence
                await self.server_ws.send(json.dumps({
                    'type': 'audio',
                    'data': base64.b64encode(dummy_audio).decode('utf-8')
                }))
                logger.info("✅ Sent initial trigger to AI server")
        except Exception as e:
            logger.error(f"Error triggering initial greeting: {e}")

    async def handle_server_messages(self):
        try:
            async for message in self.server_ws:
                if not self.stream_active:
                    break

                data = json.loads(message)
                logger.info(f"📨 Received from AI server: {data.get('type')}")

                if data['type'] == 'tts_file':
                    await self.play_tts_audio(data['filepath'], data['duration'], data['text'])
                elif data['type'] == 'transcription':
                    logger.info(f"🎤 Transcription: {data['text']}")
                elif data['type'] == 'status':
                    logger.info(f"📊 Status: {data.get('message')}")

        except Exception as e:
            logger.error(f"Error handling server messages: {e}")
            self.is_connected = False

    async def play_tts_audio(self, filepath, duration, text):
        """Play TTS audio without interrupting the stream"""
        try:
            logger.info(f"🔊 Playing TTS: {text}")

            # Ensure static directory exists
            static_dir = os.path.join(os.getcwd(), 'static', 'audio')
            os.makedirs(static_dir, exist_ok=True)

            # Copy file to static directory
            filename = f"tts_{int(time.time() * 1000)}.wav"
            static_path = os.path.join(static_dir, filename)
            shutil.copy2(filepath, static_path)

            # Create new TwiML with audio and continued streaming
            response = VoiceResponse()
            response.play(f"{NGROK_URL}/static/audio/{filename}")

            # CRITICAL: Restart the stream after audio
            start = Start()
            start.stream(
                url=f'wss://{NGROK_URL.replace("https://", "").replace("http://", "")}/stream'
            )
            response.append(start)
            response.pause(length=3600)  # Keep alive

            # Update the call
            call = self.twilio_client.calls(self.call_sid)
            call.update(twiml=str(response))

            logger.info(f"✅ Updated call with TTS audio: {filename}")

            # Wait for audio to finish
            await asyncio.sleep(duration + 1)

        except Exception as e:
            logger.error(f"❌ Error playing TTS: {e}")
        finally:
            # Cleanup
            try:
                if os.path.exists(static_path):
                    os.unlink(static_path)
                if os.path.exists(filepath):
                    os.unlink(filepath)
            except:
                pass

    async def process_audio(self, audio_data):
        """Process audio and send to AI server"""
        if not self.is_connected or not self.server_ws:
            if not self.first_message_sent:
                logger.warning("⚠️  WebSocket not connected, cannot process audio")
                self.first_message_sent = True
            return

        try:
            processed_audio = self.audio_processor.process_mulaw(audio_data)
            if processed_audio:
                await self.server_ws.send(json.dumps({
                    'type': 'audio',
                    'data': base64.b64encode(processed_audio).decode('utf-8')
                }))

        except Exception as e:
            logger.error(f"❌ Error processing audio: {e}")


@app.route("/voice", methods=['POST'])
def voice():
    """Handle incoming calls with immediate stream setup"""
    try:
        call_sid = request.values.get('CallSid')
        from_number = request.values.get('From')
        logger.info(f"📞 New call from {from_number}, SID: {call_sid}")

        # Store call state
        call_states[call_sid] = {
            'status': 'initiated',
            'start_time': time.time()
        }

        response = VoiceResponse()

        # Immediate stream setup without Say command
        start = Start()
        start.stream(
            url=f'wss://{NGROK_URL.replace("https://", "").replace("http://", "")}/stream'
        )
        response.append(start)

        # Keep call alive
        response.pause(length=3600)

        # Create bridge and start connection immediately
        bridge = WebSocketBridge(call_sid)
        active_calls[call_sid] = bridge

        def start_websocket():
            asyncio.set_event_loop(bridge.loop)
            try:
                bridge.loop.run_until_complete(bridge.connect_to_server())
                bridge.loop.run_forever()
            except Exception as e:
                logger.error(f"WebSocket loop error: {e}")

        Thread(target=start_websocket, daemon=True).start()

        logger.info(f"✅ Call setup complete for {call_sid}")
        return str(response)

    except Exception as e:
        logger.error(f"❌ Error in voice webhook: {e}")
        return str(VoiceResponse())


@sock.route('/stream')
def stream(ws):
    """Handle Twilio media stream with better error handling"""
    bridge = None
    call_sid = None

    try:
        logger.info("🔌 New WebSocket connection")

        # Handle connection setup
        connect_msg = json.loads(ws.receive())
        logger.info(f"📥 Connect message: {connect_msg}")

        start_msg = json.loads(ws.receive())
        logger.info(f"🚀 Start message: {start_msg}")

        if start_msg.get('event') == 'start':
            call_sid = start_msg.get('start', {}).get('callSid')
            logger.info(f"📞 Stream started for call: {call_sid}")

            # Wait for bridge to be ready
            max_wait = 10  # 10 seconds
            wait_time = 0
            while call_sid not in active_calls and wait_time < max_wait:
                time.sleep(0.5)
                wait_time += 0.5

            bridge = active_calls.get(call_sid)

            if not bridge:
                logger.error(f"❌ No bridge found for call {call_sid}")
                return

            logger.info(f"✅ Bridge connected for call {call_sid}")

            # Update call state
            if call_sid in call_states:
                call_states[call_sid]['status'] = 'streaming'

            # Process media stream
            while bridge.stream_active:
                try:
                    message = ws.receive()
                    if not message:
                        continue

                    data = json.loads(message)

                    if data.get('event') == 'media':
                        media = data.get('media', {})
                        if media.get('track') == 'inbound':
                            payload = media.get('payload')
                            if payload:
                                audio_data = base64.b64decode(payload)
                                # Send to bridge asynchronously
                                asyncio.run_coroutine_threadsafe(
                                    bridge.process_audio(audio_data),
                                    bridge.loop
                                )

                    elif data.get('event') == 'stop':
                        logger.info(f"🛑 Stream stopped for call {call_sid}")
                        bridge.stream_active = False
                        break

                except Exception as e:
                    if "WebSocket connection is closed" in str(e):
                        logger.info(f"🔌 WebSocket closed for call {call_sid}")
                        break
                    else:
                        logger.error(f"❌ Stream processing error: {e}")

    except Exception as e:
        logger.error(f"❌ Stream handler error: {e}")
    finally:
        if bridge:
            bridge.stream_active = False
        if call_sid and call_sid in call_states:
            call_states[call_sid]['status'] = 'ended'


@app.route("/status", methods=['POST'])
def status():
    """Handle call status updates"""
    try:
        call_sid = request.values.get('CallSid')
        call_status = request.values.get('CallStatus')

        logger.info(f"📊 Call {call_sid} status: {call_status}")

        if call_sid in call_states:
            call_states[call_sid]['status'] = call_status

        if call_status in ['completed', 'failed', 'busy', 'no-answer', 'canceled']:
            if call_sid in active_calls:
                bridge = active_calls[call_sid]
                bridge.stream_active = False
                del active_calls[call_sid]

            if call_sid in call_states:
                del call_states[call_sid]

        return '', 200
    except Exception as e:
        logger.error(f"❌ Status webhook error: {e}")
        return '', 500


@app.route("/debug")
def debug():
    """Debug endpoint to check system status"""
    return jsonify({
        'active_calls': len(active_calls),
        'call_states': call_states,
        'server_running': True
    })


@app.route('/static/audio/<filename>')
def serve_audio(filename):
    """Serve audio files"""
    return send_from_directory('static/audio', filename)


def start_client():
    """Make outbound call"""
    try:
        time.sleep(3)  # Wait for server to be ready

        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        logger.info(f"📞 Making call to {TO_NUMBER}")

        call = client.calls.create(
            to=TO_NUMBER,
            from_=TWILIO_PHONE_NUMBER,
            url=f"{NGROK_URL}/voice",
            status_callback=f"{NGROK_URL}/status",
            status_callback_event=['initiated', 'answered', 'completed'],
            timeout=120,
            record=False
        )

        logger.info(f"✅ Call initiated: {call.sid}")
        return call.sid

    except Exception as e:
        logger.error(f"❌ Error making call: {e}")
        return None


if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs('static/audio', exist_ok=True)

    logger.info("🚀 Starting Twilio bridge server...")

    # Start call in background
    Thread(target=start_client, daemon=True).start()

    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)
